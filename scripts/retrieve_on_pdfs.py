from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from vidore_benchmark.evaluation.evaluate import get_top_k
from vidore_benchmark.retrievers.utils.load_retriever import load_vision_retriever_from_registry
from vidore_benchmark.utils.image_utils import generate_dataset_from_img_folder
from vidore_benchmark.utils.pdf_utils import convert_all_pdfs_to_images

load_dotenv(override=True)


def main(
    model_name: Annotated[str, typer.Option(help="Model name to use for evaluation")],
    query: Annotated[str, typer.Option(help="Query to use for retrieval")],
    k: Annotated[int, typer.Option(help="Number of documents to retrieve")],
    data_dirpath: Annotated[
        str, typer.Option(help="Path to the folder containing the PDFs to use as the retrieval corpus")
    ],
    batch_size: Annotated[int, typer.Option(help="Batch size for document embedding inference")] = 4,
):
    """
    This script is used to ask a query and retrieve the top-k documents from a given folder containing PDFs.
    The PDFs will be converted to a dataset of image pages and then used for retrieval.
    """

    assert Path(data_dirpath).is_dir(), f"Invalid data directory: `{data_dirpath}`"

    # Create the vision retriever
    retriever = load_vision_retriever_from_registry(model_name)

    # Convert the PDFs to a collection of images
    convert_all_pdfs_to_images(data_dirpath)
    image_files = list(Path(data_dirpath).rglob("*.jpg"))
    print(f"Found {len(image_files)} images in the directory `{data_dirpath}`")

    # Generate a dataset using the images
    dataset = generate_dataset_from_img_folder(data_dirpath)

    # Get the top-k documents
    top_k = get_top_k(
        retriever,
        queries=[query],
        documents=list(dataset["image"]),
        file_names=list(dataset["image_filename"]),
        batch_query=1,
        batch_doc=batch_size,
        k=k,
    )

    print(f"Top-{k} documents for the query '{query}':")

    for document, score in top_k[query].items():  # type: ignore
        print(f"Document: {document}, Score: {score}")


if __name__ == "__main__":
    typer.run(main)
