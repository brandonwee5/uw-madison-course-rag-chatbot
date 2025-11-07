from llama_parse import LlamaParse
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


parser = LlamaParse(
   result_type="markdown",  # "markdown" and "text" are available
   )

file_name = "report-gradedistribution-2024-2025spring.pdf"
extra_info = {"file_name": file_name}

with open(f"./{file_name}", "rb") as f:
   # must provide extra_info with file_name key with passing file object
   documents = parser.load_data(f, extra_info=extra_info)
   print(f"Total chunks parsed: {len(documents)}")

output_file = "fullgrades.md"
with open(output_file, "w", encoding="utf-8") as f:
    total = len(documents)
    for i, doc in enumerate(documents, 1):
        f.write(doc.text + "\n\n")  # add spacing between chunks
        # Print progress every 10 chunks or at the end
        if i % 10 == 0 or i == total:
            print(f"Processed {i}/{total} chunks ({i/total*100:.1f}%)")

print(f"\n✅ Parsing complete! Output saved to {output_file}")