#### TABLES

nohup python3 -u code-syntax-concept.py > logs/gpt-neo-125M.txt &
nohup python3 -u code-syntax-concept.py > logs/gpt-neo-1.3B.txt &
nohup python3 -u code-syntax-concept.py > logs/gpt-neox-20b.txt &

nohup python3 -u code-syntax-concept.py > logs/codegpt-small-py-adaptedGPT2.txt &
nohup python3 -u code-syntax-concept.py > logs/codegpt-small-py.txt &

nohup python3 -u code-syntax-concept.py > logs/pycoder-gpt2.txt &

# Salesforce
nohup python3 -u code-syntax-concept.py > logs/codegen-16B-multi.txt &
nohup python3 -u code-syntax-concept.py > logs/codegen-6B-multi.txt &
nohup python3 -u code-syntax-concept.py > logs/codegen-2B-multi.txt &


### Agregation Function
nohup python3 -u aggregation-function.py > logs/aggregation_function_gpt-neo-1.3B.txt &
nohup python3 -u aggregation-function.py > logs/aggregation_function_gpt-neo-2.7B.txt &
nohup python3 -u aggregation-function.py > logs/aggregation_function_codegen-2B-nl.txt &

### Embeddings 
nohup python3 -u embedding-function.py > logs/embedding_function_gpt-neo-1.3B.txt &
nohup python3 -u embedding-function.py > logs/embedding_function_gpt-neo-2.7B.txt &
nohup python3 -u embedding-function.py > logs/embedding_function_codegen-2B-nl.txt &

#### Global 
nohup python3 -u global-aggregates.py > logs/global-aggregates_gpt-neo-1.3B.txt &
nohup python3 -u global-aggregates.py > logs/global-aggregates_gpt-neo-2.7B.txt &
nohup python3 -u global-aggregates.py > logs/global-aggregates_codegen-2B-nl.txt &