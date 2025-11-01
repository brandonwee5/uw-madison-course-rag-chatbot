from llama_parse import LlamaParse


parser = LlamaParse(
   result_type="markdown",  # "markdown" and "text" are available
   )

file_name = "gradedistribution_first5pages.pdf"
extra_info = {"file_name": file_name}

with open(f"./{file_name}", "rb") as f:
   # must provide extra_info with file_name key with passing file object
   documents = parser.load_data(f, extra_info=extra_info)

with open('output.md', 'w') as f:
   print(documents, file=f)

# Write the output to a file
with open("output.md", "w", encoding="utf-8") as f:
   for doc in documents:
       f.write(doc.text)