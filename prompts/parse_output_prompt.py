sys_prompt = """
You are familar with the MARKDOWN syntax and expression.
You need to parse a MARKDOWN file with provided information.
The format for the file is:

# FAULT DIAGNOSIS REPORT

## SYSTEM INFORMATION
fault type:
fault level:
fault refinement:
fault description:

## SEARCH QUERY
show queries, provided in the USER PROMPT, in bullet points in this section

## SEARCH RESULTS
show search results, provided in the USER PROMPT, in bullet points in this section

## RECOMMENDATIONS 

## CONCLUSION
You should summarize the all mentioned content of the file in a concise and accurate way as a conclusion.
"""

user_prompt = """
USER PROMPT:
Please parse a markdown file with following texts:
fault: {fault_text}
query: {query_text}
answer: {answer_text}
rec: {rec_text}
"""