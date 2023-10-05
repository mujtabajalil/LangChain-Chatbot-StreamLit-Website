from langchain.chains import VectorDBQAWithSourcesChain
import pickle
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
import streamlit as st
from streamlit_chat import message as st_message
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.llm import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
import json
import openai

template = """
You are a helpful assistant providing guidance on how to perform specific actions on a website or platform.

To {action}, please follow these general steps:

1. Visit the official website or platform where you have your online account.

2. Sign in to your online account at {website_url}.

3. {additional_steps}

4. {specific_instructions}

5. {important_note}

If you encounter any difficulties during this process, please refer to the website's or platform's support resources for further assistance.

For more detailed information, you can visit the relevant section on the website: {more_information_url}.
"""

prompt = PromptTemplate(template=template,
                         input_variables=["action", "website_url","additional_steps","specific_instructions",
                                          "important_note","more_information_url"])


llm = OpenAI(openai_api_key="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")

def load_vector_store():
    # Indexing
    ### Load vector store
    with open("faiss_store.pkl", "rb") as f:
        vectordb = pickle.load(f)

    return vectordb



chat_history = []
vectordb = load_vector_store()



chatgpt_chain = LLMChain(prompt=prompt, 
                         llm=llm)



# Load QA chain
doc_chain = load_qa_with_sources_chain(llm, chain_type="stuff")

# Create Conversational Retrieval Chain
chain = ConversationalRetrievalChain(
    retriever=vectordb.as_retriever(),
    combine_docs_chain=doc_chain,
    question_generator=chatgpt_chain,
    return_source_documents=True
)





# Set up the Streamlit app
st.title("GOV.UK")
st.markdown(
    """ 
    ####  🗨️ Student Finance England 
    """
)

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything about this website 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] =  ["Hey ! 👋"]

#container for the chat history and user's text input
response_container, container = st.container(), st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        # Allow the user to enter a query and generate a response
        user_input  = st.text_input(
            "**Talk with your website here**",
            placeholder="Talk with your website here.",
        )
        submit_button = st.form_submit_button(label='Send')

        if user_input:
            with st.spinner(
                "Generating Answer to your Query : `{}` ".format(user_input )
            ):
                result = chain({"question": user_input, 
                    "chat_history": chat_history})

                ans = result['answer']


                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(ans)

if st.session_state['generated']:
   with response_container:
       for i in range(len(st.session_state['generated'])):
           st_message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="fun-emoji")
           st_message(st.session_state["generated"][i], key=str(i), avatar_style="croodles-neutral")