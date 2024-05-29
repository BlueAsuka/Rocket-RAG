sys_prompt = """
You are familar with the MARKDOWN syntax and expression.
You need to parse a MARKDOWN file and store it in a specific directory given by user.
The format for the file is:

# FAULT DIAGNOSIS REPORT

## SYSTEM INFORMATION
fault type:
fault level:
fault description:

## SEARCH QUERY

## SEARCH DIAGNOSIS

## RECOMMENDATIONS 
"""

user_prompt = """
Please parse a markdown file with following texts:
fault: {fault_text}
query: {query_text}
answer: {answer_text}
rec: {rec_text}
store the file in the path {file_path}
"""