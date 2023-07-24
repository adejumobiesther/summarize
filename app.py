from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI, LLMChain, HuggingFaceHub
import textwrap


def summarize(doc):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0,
                                          separator="\n")  # Initilize an instance of CharacterTextSplitter
    chunks = text_splitter.split_text(doc)
    doc_store = [Document(page_content=text) for text in chunks]
    llm_model = OpenAI(model_name="text-davinci-003", temperature=0)  # define your language model

    summarization_chain2 = load_summarize_chain(llm=llm_model,
                                                chain_type='map_reduce',
                                                verbose=True  # define the chain type)
                                                )
    output_summary = summarization_chain2.run(doc_store)
    wrapped_text = textwrap.fill(output_summary, width=100)

    return wrapped_text

if __name__ == "__main__":
    # make a gradio interface
    import gradio as gr

    outputs = gr.outputs.Textbox()

    app = gr.Interface(fn=summarize, inputs='text', outputs=outputs,description="This is a text summarization model").launch()














