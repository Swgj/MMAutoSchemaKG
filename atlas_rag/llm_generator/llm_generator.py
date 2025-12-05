import json
import asyncio
from openai import OpenAI, AzureOpenAI, NOT_GIVEN
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_fixed, wait_exponential, wait_random, RetryCallState
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from atlas_rag.llm_generator.prompt.rag_prompt import cot_system_instruction, cot_system_instruction_kg, cot_system_instruction_no_doc, prompt_template
from atlas_rag.llm_generator.prompt.lkg_prompt import ner_prompt, keyword_filtering_prompt, simple_ner_prompt
from atlas_rag.llm_generator.prompt.rag_prompt import filter_triple_messages
from atlas_rag.llm_generator.format.validate_json_output import *
from atlas_rag.llm_generator.format.validate_json_schema import filter_fact_json_schema, lkg_keyword_json_schema, ATLAS_SCHEMA
from transformers.pipelines import Pipeline
import jsonschema
import time
from transformers import AutoTokenizer



def serialize_openai_tool_call_message(message) -> dict:
    # Initialize the output dictionary
    serialized = {
        "role": message.role,
        "content": None if not message.content else message.content,
        "tool_calls": []
    }
    
    # Serialize each tool call
    for tool_call in message.tool_calls:
        serialized_tool_call = {
            "id": tool_call.id,
            "type": tool_call.type,
            "function": {
                "name": tool_call.function.name,
                "arguments": json.dumps(tool_call.function.arguments)
            }
        }
        serialized["tool_calls"].append(serialized_tool_call)
    
    return serialized

def print_retry(retry_state: RetryCallState):
    # Access the instance via retry_state.args[0] if method is bound
    instance = retry_state.args[0] if retry_state.args else None
    if instance and hasattr(instance, 'retry_count'):
        instance.retry_count += 1
        print(f"Retrying {retry_state.fn.__name__}: attempt {retry_state.attempt_number}, total retries: {instance.retry_count}")
    else:
        print(f"Retrying {retry_state.fn.__name__}: attempt {retry_state.attempt_number}")
retry_decorator = retry(
    stop=(stop_after_delay(120) | stop_after_attempt(5)),  # Max 2 minutes or 5 attempts
    # wait=wait_exponential(multiplier=1, min=2, max=30) + wait_random(min=0, max=2),
    wait=wait_fixed(0),
    before_sleep=print_retry,
)
class LLMGenerator():
    def __init__(self, client, model_name, backend='openai', max_workers=8):
        self.model_name = model_name
        self.client : OpenAI|Pipeline  = client
        self.max_workers = max_workers
        if isinstance(client, OpenAI|AzureOpenAI):
            self.inference_type = "openai"
        elif isinstance(client, Pipeline):
            self.inference_type = "pipeline"
        else:
            raise ValueError("Unsupported client type. Please provide either an OpenAI client or a Huggingface Pipeline Object.")
        
        if backend == 'vllm':
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.retry_count = 0

    @retry_decorator
    def _api_inference(self, message, max_new_tokens=8192,
                           temperature = 0.7,
                           frequency_penalty = None,
                           response_format = {"type": "text"},
                           return_text_only=True,
                           return_thinking=False,
                           reasoning_effort=None,
                           **kwargs):
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    frequency_penalty= NOT_GIVEN if frequency_penalty is None else frequency_penalty,
                    response_format = response_format if response_format is not None else {"type": "text"},
                    timeout = 120,
                    reasoning_effort= NOT_GIVEN if reasoning_effort is None else reasoning_effort,
                    # extra_body = {
                    #     "chat_template_kwargs": {"enable_thinking": False if reasoning_effort is None else True}
                    # }
                    
                )
        except Exception as e:
            print(f"API inference failed: {e}")
            raise e
        time_cost = time.time() - start_time
        content = response.choices[0].message.content
        if content is None and hasattr(response.choices[0].message, 'reasoning_content'):
            content = response.choices[0].message.reasoning_content
        validate_function = kwargs.get('validate_function', None)
        content = validate_function(content, **kwargs) if validate_function else content

        if '</think>' in content and not return_thinking:
            content = content.split('</think>')[-1].strip()
        else:
            if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content is not None and return_thinking:
                content = '<think>' + response.choices[0].message.reasoning_content + '</think>' + content
        

        if return_text_only:
            return content
        else:
            completion_usage_dict = response.usage.model_dump()
            completion_usage_dict['time'] = time_cost
            return content, completion_usage_dict

    def generate_response(self, batch_messages, do_sample=True, max_new_tokens=8192,
                                 temperature=0.7, frequency_penalty=None, response_format={"type": "text"},
                                 return_text_only=True, return_thinking=False, reasoning_effort=None, **kwargs):
        if temperature == 0.0:
            do_sample = False
        # single = list of dict, batch = list of list of dict
        is_batch = isinstance(batch_messages[0], list)
        if not is_batch:
            batch_messages = [batch_messages]
        batch_size = len(batch_messages)
        results = [None] * len(batch_messages)
        to_process = list(range(len(batch_messages)))
        if self.inference_type == "openai":
            max_workers = self.max_workers  # Default to 4 workers if not specified
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                def process_message(i):
                    try:
                        return self._api_inference(
                            batch_messages[i], max_new_tokens, temperature,
                            frequency_penalty, response_format, return_text_only, return_thinking, reasoning_effort, **kwargs
                        )
                    except Exception as e:
                        return "[]"
                futures = [executor.submit(process_message, i) for i in to_process]
            for i, future in enumerate(futures):
                try:
                    results[i] = future.result()
                except Exception as e:
                    print(f"Future {i} failed: {str(e)}")
                    results[i] = '[]'  # Fallback to empty list on failure

        elif self.inference_type == "pipeline":
            max_retries = kwargs.get('max_retries', 3)  # Default to 3 retries if not specified
            start_time = time.time()
            # Initial processing of all messages
            responses = self.client(
                batch_messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                return_full_text=False
            )
            time_cost = time.time() - start_time
            
            # Extract contents
            contents = [resp[0]['generated_text'].strip() for resp in responses]
            
            # Validate and collect failed indices
            validate_function = kwargs.get('validate_function', None)
            failed_indices = []
            for i, content in enumerate(contents):
                if validate_function:
                    try:
                        contents[i] = validate_function(content, **kwargs)
                    except Exception as e:
                        print(f"Validation failed for index {i}: {e}")
                        failed_indices.append(i)
            
            # Retry failed messages in batches
            for attempt in range(max_retries):
                if not failed_indices:
                    break  # No more failures to retry
                print(f"Retry attempt {attempt + 1}/{max_retries} for {len(failed_indices)} failed messages")
                # Prepare batch of failed messages
                failed_messages = [batch_messages[i] for i in failed_indices]
                try:
                    # Process failed messages as a batch
                    retry_responses = self.client(
                        failed_messages,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample,
                        return_full_text=False
                    )
                    retry_contents = [resp[0]['generated_text'].strip() for resp in retry_responses]
                    
                    # Validate retry results and update contents
                    new_failed_indices = []
                    for j, i in enumerate(failed_indices):
                        try:
                            if validate_function:
                                retry_contents[j] = validate_function(retry_contents[j], **kwargs)
                            contents[i] = retry_contents[j]
                        except Exception as e:
                            print(f"Validation failed for index {i} on retry {attempt + 1}: {e}")
                            new_failed_indices.append(i)
                    failed_indices = new_failed_indices  # Update failed indices for next retry
                except Exception as e:
                    print(f"Batch retry {attempt + 1} failed: {e}")
                    # If batch processing fails, keep all indices in failed_indices
                    if attempt == max_retries - 1:
                        for i in failed_indices:
                            contents[i] = ""  # Set to "" if all retries fail
            
            # Set remaining failed messages to "" after all retries
            for i in failed_indices:
                contents[i] = ""
            
            # Process thinking tags
            if not return_thinking:
                contents = [content.split('</think>')[-1].strip() if '</think>' in content else content for content in contents]
            
            if return_text_only:
                results = contents
            else:
                usage_dicts = [{
                    'completion_tokens': len(content.split()),
                    'time': time_cost / len(batch_messages)
                } for content in contents]
                results = list(zip(contents, usage_dicts))
        return results[0] if not is_batch else results

    def generate_cot(self, question, max_new_tokens=1024):
        messages = [
            {"role": "system", "content": "".join(cot_system_instruction_no_doc)},
            {"role": "user", "content": question},
        ]
        return self.generate_response(messages, max_new_tokens=max_new_tokens)

    def generate_with_context(self, question, context, max_new_tokens=1024, temperature = 0.7):
        messages = [
            {"role": "system", "content": "".join(cot_system_instruction)},
            {"role": "user", "content": f"{context}\n\n{question}\nThought:"},
        ]
        return self.generate_response(messages, max_new_tokens=max_new_tokens, temperature = temperature)

    def generate_with_context_one_shot(self, question, context, max_new_tokens=4096, temperature = 0.7):
        messages = deepcopy(prompt_template)
        messages.append(
            {"role": "user", "content": f"{context}\n\nQuestions:{question}\nThought:"},
        )
        return self.generate_response(messages, max_new_tokens=max_new_tokens, temperature = temperature)
    
    def generate_with_context_kg(self, question, context, max_new_tokens=1024, temperature = 0.7):
        messages = [
            {"role": "system", "content": "".join(cot_system_instruction_kg)},
            {"role": "user", "content": f"{context}\n\n{question}"},
        ]
        return self.generate_response(messages, max_new_tokens=max_new_tokens, temperature = temperature)
        
    @retry_decorator
    def filter_triples_with_entity_event(self,question, triples):
        messages = deepcopy(filter_triple_messages)
        messages.append(
            {"role": "user", "content": f"""[ ## question ## ]]
        {question}

        [[ ## fact_before_filter ## ]]
        {triples}"""})
        try:
            validate_args = {
                "schema": filter_fact_json_schema,
                "fix_function": fix_filter_triplets,
            }
            response = self.generate_response(messages, max_new_tokens=4096, temperature=0.0, response_format={"type": "json_object"},
                                               validate_function=validate_output, **validate_args)
            return response
        except Exception as e:
            # If all retries fail, return the original triples
            return triples
        
    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(2))
    def large_kg_filter_keywords_with_entity(self, question, keywords):
        messages = deepcopy(keyword_filtering_prompt)
        
        messages.append({
            "role": "user",
            "content": f"""[[ ## question ## ]]
            {question}
            [[ ## keywords_before_filter ## ]]
            {keywords}"""
        })
        
        try:
            response = self.generate_response(messages, response_format={"type": "json_object"}, temperature=0.0, max_new_tokens=2048)
            
            # Validate and clean the response
            cleaned_data = validate_output(response, lkg_keyword_json_schema, fix_lkg_keywords)
            
            return cleaned_data['keywords']
        except Exception as e:
            return keywords
    
    def ner(self, text):
        messages = [
            {"role": "system", "content": "Please extract the entities from the following question and output them separated by comma, in the following format: entity1, entity2, ..."},
            {"role": "user", "content": f"Extract the named entities from: {text}"},
        ]
        return self.generate_response(messages)
    
    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(2))
    def large_kg_ner(self, text, simple_ner = False):
        if not simple_ner:
            messages = deepcopy(ner_prompt)
            messages.append(
                {
                    "role": "user", 
                    "content": f"[[ ## question ## ]]\n{text}" 
                }
            )
        else:
            messages = deepcopy(simple_ner_prompt)
            messages.append(
                {
                    "role": "user", 
                    "content": """
                    extracts named entities from given text.
                    Output them in Json format as follows:
                    {
                        "keywords": ["entity1", "entity2", ...]
                    }
                    Given text: 
                    """ + text
                }
            )
        validation_args = {
            "schema": lkg_keyword_json_schema,
            "fix_function": fix_lkg_keywords
        }
        # Generate raw response from LLM
        raw_response = self.generate_response(messages, max_new_tokens=4096, temperature=0.7, frequency_penalty=1.1, response_format={"type": "json_object"}, validate_output=validate_output, **validation_args)
        
        try:
            # Validate and clean the response
            cleaned_data = json_repair.loads(raw_response)
            return cleaned_data['keywords']
        
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            return []  # Fallback to empty list or raise custom exception
 
    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(2))
    def large_kg_tog_ner(self, text):
        messages = deepcopy(simple_ner_prompt)
        messages.append(
            {
                "role": "user", 
                "content": """
                extracts named entities from given text.
                Output them in Json format as follows:
                {
                    "keywords": ["entity1", "entity2", ...]
                }
                Given text: 
                """ + text
            }
        )
        # Generate raw response from LLM
        validation_args = {
            "schema": lkg_keyword_json_schema,
            "fix_function": fix_lkg_keywords
        }
        raw_response = self.generate_response(messages, max_new_tokens=4096, temperature=0.7, frequency_penalty=1.1, response_format={"type": "json_object"}, validate_output=validate_output, **validation_args)
        
        try:
            # Validate and clean the response
            cleaned_data = json_repair.loads(raw_response)
            return cleaned_data['keywords']
        
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            return []  # Fallback to empty list or raise custom exception

    def generate_with_react(self, question, context=None, max_new_tokens=1024, search_history=None, logger=None):
        react_system_instruction = (
            'You are an advanced AI assistant that uses the ReAct framework to solve problems through iterative search. '
            'Follow these steps in your response:\n'
            '1. Thought: Think step by step and analyze if the current context is sufficient to answer the question. If not, review the current context and think critically about what can be searched to help answer the question.\n'
            '   - Break down the question into *1-hop* sub-questions if necessary (e.g., identify key entities like people or places before addressing specific events).\n'
            '   - Use the available context to make inferences about key entities and their relationships.\n'
            '   - If a previous search query (prefix with "Previous search attempt") was not useful, reflect on why and adjust your strategy—avoid repeating similar queries and consider searching for general information about key entities or related concepts.\n'
            '2. Action: Choose one of:\n'
            '   - Search for [Query]: If you need more information, specify a new query. The [Query] must differ from previous searches in wording and direction to explore new angles.\n'
            '   - No Action: If the current context is sufficient.\n'
            '3. Answer: Provide one of:\n'
            '   - A concise, definitive response as a noun phrase if you can answer.\n'
            '   - "Need more information" if you need to search.\n\n'
            'Format your response exactly as:\n'
            'Thought: [your reasoning]\n'
            'Action: [Search for [Query] or No Action]\n'
            'Answer: [concise noun phrase if you can answer, or "Need more information" if you need to search]\n\n'
        )
        
        # Build context with search history if available
        full_context = []
        if search_history:
            for i, (thought, action, observation) in enumerate(search_history):
                search_history_text = f"\nPrevious search attempt {i}:\n"
                search_history_text += f"{action}\n  Result: {observation}\n"
                full_context.append(search_history_text)
        if context:
            full_context_text = f"Current Retrieved Context:\n{context}\n"
            full_context.append(full_context_text)
        if logger:
            logger.info(f"Full context for ReAct generation: {full_context}")
        
        # Combine few-shot examples with system instruction and user query
        messages = [
            {"role": "system", "content": react_system_instruction},
            {"role": "user", "content": f"Search History:\n\n{''.join(full_context)}\n\nQuestion: {question}" 
            if full_context else f"Question: {question}"}
        ]
        if logger:
            logger.info(f"Messages for ReAct generation: {search_history}Question: {question}")
        return self.generate_response(messages, max_new_tokens=max_new_tokens)

    
    def triple_extraction(self, messages, result_schema, max_tokens=4096, stage=None, record=False, allow_empty=True):
        if isinstance(messages[0], dict):
            messages = [messages]
        validate_kwargs = {
            'schema': result_schema,
            'fix_function': fix_triple_extraction_response, # modify the fix according to provided schema
            'allow_empty': allow_empty
        }
        try:
            result = self.generate_response(messages, max_new_tokens=max_tokens, validate_function=validate_output, return_text_only = not record, **validate_kwargs)
            return result
        except Exception as e:
            print(f"Triple extraction failed: {e}")
            # Return empty result if validation fails and allow_empty is True
            if allow_empty:
                if record:
                    return [], {'completion_tokens': 0, 'time': 0}
                else:
                    return []
            else:
                raise e