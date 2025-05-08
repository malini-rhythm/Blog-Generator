import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
import time

def get_llam_response(input_text,no_words,blog_style,model_size):

    #llm models 
    m1_4GB = "D:\DS PRO\PROJECTS\llama models\llama-2-7b-chat.ggmlv3.q4_1.bin"
    m2_2GB = "D:\DS PRO\PROJECTS\llama models\llama-2-7b-chat.ggmlv3.q2_K.bin"

    model_chosen = m1_4GB if model_size=="4.33GB" else m2_2GB
    model = CTransformers(model=model_chosen,model_type='llama',config={'max_new_tokens':256,'temperature':0.01})
    
    #Prompt template
    template = """
    write a blog on topic {input_text} assuming the reader is {blog_style}
    within {no_words} words.
    """
    prompt = PromptTemplate(input_variables=["blog_style","input_text","no_words"],template=template)
    
    #Generate response from the llama model
    response=model(prompt.format(blog_style=blog_style,input_text=input_text,no_words=no_words))
    
    print(response)
    return response

st.set_page_config(
    page_title = "Blog Generator",
    layout = "centered",
)

st.header("BLOG GENERATOR")
title = st.text_input("Enter Blog Title")

#column creation
col1,col2,col3 = st.columns([5,5,5])
with col1:
    word_count = st.text_input("Word Limit")
with col2:
    audience = st.selectbox("Target Audience",('Resercher','Data Scientist','Common People'),index = 0)
with col3:
    model_size = st.selectbox("Choose Model Size",('2.5GB model','4.33GB'),index = 0)

submit = st.button("Generate")

if submit :
    start_time = time.time()
    st.write(get_llam_response(title,word_count,audience,model_size))
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Time Taken: {elapsed_time} seconds")
    st.write(f"Time Taken: {elapsed_time} seconds")

                                             