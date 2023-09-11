from transformers import PreTrainedTokenizerFast, PreTrainedTokenizerFast, BartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import ElectraTokenizer, pipeline,AutoModelForSeq2SeqLM, ElectraForQuestionAnswering
from transformers import AutoTokenizer, AutoModelForSequenceClassification, ElectraConfig
from emotion.multilabel_pipeline import MultiLabelPipeline
from emotion.model import ElectraForMultiLabelClassification
from care.koelectra import koElectraForSequenceClassification
import torch

print("loading...")
to = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#qna_token = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-finetuned-korquad") 
#qna_model = ElectraForQuestionAnswering.from_pretrained("monologg/koelectra-base-v3-finetuned-korquad") 
qa = "monologg/koelectra-base-v3-finetuned-korquad"
qa_func = pipeline("question-answering", tokenizer=qa, model=qa, device=0) 

print("qa model loaded")

#senti_token = AutoTokenizer.from_pretrained("circulus/koelectra-sentiment-v1") #torch_dtype=torch.float16
#senti_model = AutoModelForSequenceClassification.from_pretrained("circulus/koelectra-sentiment-v1") #torch_dtype=torch.float16
senti = "circulus/koelectra-sentiment-v1"
senti_func = pipeline("text-classification", tokenizer=senti, model=senti, device=-1) 

#polite_token = AutoTokenizer.from_pretrained("circulus/koelectra-polite-v1") # ,torch_dtype=torch.float16)
#polite_model = AutoModelForSequenceClassification.from_pretrained("circulus/koelectra-polite-v1") #,torch_dtype=torch.float16)

polite = "circulus/koelectra-polite-v1"
polite_func = pipeline("text-classification", tokenizer=polite, model=polite, device=-1) 

#grammer_token = AutoTokenizer.from_pretrained("circulus/koelectra-polite-v1") #,torch_dtype=torch.float16)
#grammer_model = AutoModelForSequenceClassification.from_pretrained("circulus/koelectra-polite-v1") #,torch_dtype=torch.float16)
grammer = "circulus/koelectra-polite-v1"
grammer_func = pipeline("text-classification", tokenizer=grammer, model=grammer, device=-1) 

#emo_token = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-goemotions")
#emo_model = ElectraForMultiLabelClassification.from_pretrained("monologg/koelectra-base-v3-goemotions")

emo_token = ElectraTokenizer.from_pretrained("circulus/koelectra-emotion-v1") # torch_dtype=torch.float16
emo_model = ElectraForMultiLabelClassification.from_pretrained("circulus/koelectra-emotion-v1") # torch_dtype=torch.float16
emo_model.to(to)

dialect_token = ElectraTokenizer.from_pretrained("circulus/koelectra-dialect-v1") # ,torch_dtype=torch.float16
dialect_model = ElectraForMultiLabelClassification.from_pretrained("circulus/koelectra-dialect-v1") #$ ,torch_dtype=torch.float16
#dialect_model.to(to)

#hate_token = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-hate-speech")
#hate_model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-hate-speech")
#hate_func = pipeline("sentiment-analysis", tokenizer=hate_token, model=hate_model, device=device)

hate_token = ElectraTokenizer.from_pretrained("circulus/koelectra-ethics-v1",torch_dtype=torch.float16)
hate_model = ElectraForMultiLabelClassification.from_pretrained("circulus/koelectra-ethics-v1",torch_dtype=torch.float16)
hate_model.to(to)

act_token = ElectraTokenizer.from_pretrained("circulus/koelectra-act-v1") #,torch_dtype=torch.float16
act_model = ElectraForMultiLabelClassification.from_pretrained("circulus/koelectra-act-v1")# torch_dtype=torch.float16
#act_model.to(to)

well_token = ElectraTokenizer.from_pretrained("circulus/koelectra-wellness-v1") #,torch_dtype=torch.float16)
well_config = ElectraConfig.from_pretrained("circulus/koelectra-wellness-v1")
#well_config["torch_dtype"] = "torch.float16" # torch_dtype=torch.float16
#print("config",well_config)
well_model = koElectraForSequenceClassification.from_pretrained(pretrained_model_name_or_path="circulus/koelectra-wellness-v1", config=well_config, num_labels=169)
well_model.to(to)                                                             

chat_token = AutoTokenizer.from_pretrained("circulus/kobart-chat-all-v2")#,torch_dtype=torch.float16)
chat_model = AutoModelForSeq2SeqLM.from_pretrained("circulus/kobart-chat-all-v2")#,torch_dtype=torch.float16)
chat_model.to(to)

#todialect_token = PreTrainedTokenizerFast.from_pretrained('circulus/kobart-trans-dialect-v2') #,torch_dtype=torch.float16)
#todialect_model = BartForConditionalGeneration.from_pretrained('circulus/kobart-trans-dialect-v2') #,torch_dtype=torch.float16)
#todialect_model.to(to)

todialect = 'circulus/kobart-trans-dialect-v2'
todialect_func = pipeline("text2text-generation", model=todialect, tokenizer=todialect, device=0)

#tostandard_token = PreTrainedTokenizerFast.from_pretrained('circulus/kobart-trans-standard-v2')#,torch_dtype=torch.float16)
#tostandard_model = BartForConditionalGeneration.from_pretrained('circulus/kobart-trans-standard-v2')#,torch_dtype=torch.float16)
#tostandard_model.to(to)

tostandard = 'circulus/kobart-trans-standard-v2'
tostandard_func = pipeline("text2text-generation", model=tostandard, tokenizer=tostandard, device=0)

#topolite_token = PreTrainedTokenizerFast.from_pretrained('circulus/kobart-trans-polite-v2')#,torch_dtype=torch.float16)
#topolite_model = BartForConditionalGeneration.from_pretrained('circulus/kobart-trans-polite-v2')#,torch_dtype=torch.float16)
#topolite_model.to(to)
topolite = 'circulus/kobart-trans-polite-v2'
topolite_func = pipeline("text2text-generation", model=topolite, tokenizer=topolite, device=0)

#toformal_token = PreTrainedTokenizerFast.from_pretrained('circulus/kobart-trans-formal-v2')#,torch_dtype=torch.float16)
#toformal_model = BartForConditionalGeneration.from_pretrained('circulus/kobart-trans-formal-v2')#,torch_dtype=torch.float16)
#toformal_model.to(to)
toformal = 'circulus/kobart-trans-formal-v2'
toformal_func = pipeline("text2text-generation", model=toformal, tokenizer=toformal, device=0)

#toinformal_token = PreTrainedTokenizerFast.from_pretrained('circulus/kobart-trans-informal-v2')#,torch_dtype=torch.float16)
#toinformal_model = BartForConditionalGeneration.from_pretrained('circulus/kobart-trans-informal-v2')#,torch_dtype=torch.float16)
#toinformal_model.to(to)
toinformal = 'circulus/kobart-trans-informal-v2'
toinformal_func = pipeline("text2text-generation", model=toinformal, tokenizer=toinformal, device=0)

#tocorrect_token = PreTrainedTokenizerFast.from_pretrained('circulus/kobart-correct-v1')#,torch_dtype=torch.float16)
#tocorrect_model = BartForConditionalGeneration.from_pretrained('circulus/kobart-correct-v1')#,torch_dtype=torch.float16)
#tocorrect_model.to(to)
tocorrect = 'circulus/kobart-correct-v1'
tocorrect_func = pipeline("text2text-generation", model=tocorrect, tokenizer=tocorrect, device=0)

ner = "monologg/koelectra-base-v3-naver-ner"
#ner_token = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-naver-ner") #,torch_dtype=torch.float16)
#ner_model = AutoModelForTokenClassification.from_pretrained("monologg/koelectra-base-v3-naver-ner") #,torch_dtype=torch.float16)
ner_func = pipeline("ner", tokenizer=ner, model=ner, device=-1)

copywrite_token = PreTrainedTokenizerFast.from_pretrained("circulus/kobart-copywrite-v1")#,torch_dtype=torch.float16)
copywrite_model = BartForConditionalGeneration.from_pretrained("circulus/kobart-copywrite-v1")#,torch_dtype=torch.float16)
copywrite_model.to(to)

letter_token = PreTrainedTokenizerFast.from_pretrained('circulus/kobart-letter-v1')#,torch_dtype=torch.float16)
letter_model = BartForConditionalGeneration.from_pretrained('circulus/kobart-letter-v1')#,torch_dtype=torch.float16)
letter_model.to(to)

summary_token = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')#,torch_dtype=torch.float16)
summary_model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')#,torch_dtype=torch.float16)
summary_model.to(to)

#ko2en_token = PreTrainedTokenizerFast.from_pretrained('circulus/kobart-trans-ko-en-v2',torch_dtype=torch.float16)
#ko2en_model = BartForConditionalGeneration.from_pretrained('circulus/kobart-trans-ko-en-v2',torch_dtype=torch.float16)
#ko2en_model.to(to)

#en2ko_token = PreTrainedTokenizerFast.from_pretrained('circulus/kobart-trans-en-ko-v2',torch_dtype=torch.float16)
#en2ko_model = BartForConditionalGeneration.from_pretrained('circulus/kobart-trans-en-ko-v2',torch_dtype=torch.float16)
#en2ko_model.to(to)

ko2en = 'circulus/canvers-ko2en-v1'
pipe_ko2en = pipeline("text2text-generation", model=ko2en, tokenizer=ko2en, device=0)
#pipe_ko2en = pipeline("text2text-generation", model=ko2en_model, tokenizer=ko2en_token, device=0)

en2ko = 'circulus/canvers-en2ko-v1'
pipe_en2ko = pipeline("text2text-generation", model=en2ko, tokenizer=en2ko, device=0)
#pipe_en2ko = pipeline("text2text-generation", model=en2ko_model, tokenizer=en2ko_token, device=0)

print("loading ok!")