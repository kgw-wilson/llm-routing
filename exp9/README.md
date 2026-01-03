Start with random rample of about 500 prompts

Results:

embedding_type,k,accuracy,correct,total,time_sec
task_description_embedding,5,0.7,105,150,0.057965993881225586
prompt_embedding,5,0.7333333333333333,110,150,0.043555259704589844
task_description_embedding,10,0.7733333333333333,116,150,0.049471139907836914
prompt_embedding,10,0.74,111,150,0.04420804977416992
task_description_embedding,40,0.82,123,150,0.04468417167663574
prompt_embedding,40,0.7866666666666666,118,150,0.04494786262512207

Decide to do whole dataset, 

RuntimeError: len(df_filtered)=35916, len(df)=36497

A lot of effort was put into discovering the ideal prompt through testing different inputs to the task description model for different stratified samples of the dataset. Most resulted in clusters that were just about as good as prompt embeddings.

Experimented with just 1 sentence, enforced by stop tokens. Tried few shot examples. which seemed to reduce quality since model would just repeat them. Seemed like all of the data really was in the prompt not the task. Also tested different details about the task template, not a big difference from any.