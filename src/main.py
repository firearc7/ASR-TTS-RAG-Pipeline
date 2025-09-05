import hydra
from omegaconf import DictConfig
from loguru import logger
import torch
import os

from asr import transcribe_audio
from rag import create_vs
from tts import text_to_speech
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu") \
            # if not torch.backends.mps.is_available() else torch.device("mps")
    logger.info(f"Using device: {device}")

    transcribed_text = transcribe_audio(cfg.asr, device)
    logger.info(f"Transcribed Text: {transcribed_text}")

    logger.info("Initializing RAG components")
    vector_store = create_vs(
        cfg.rag.docs_path, 
        cfg.rag.vector_store_path, 
        cfg.rag.embedding_model_name,
        device)
    retriever = vector_store.as_retriever()

    llm = ChatOpenAI(
        model=cfg.llm.model, 
        base_url=cfg.llm.api_base,
        openai_api_key="nope",
    )

    prompt = ChatPromptTemplate.from_template("""Answer the following question without using any markdown, based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    logger.info("Step 4: Running RAG chain")
    response = retrieval_chain.invoke({"input": transcribed_text})
    logger.info(f"RAG Response: {response['answer']}")

    if "<think>" in response['answer']:
        response['answer'] = response['answer'].split("</think>")[1].strip()
    text_to_speech(response['answer'], cfg.tts, device)

if __name__ == "__main__":
    main()