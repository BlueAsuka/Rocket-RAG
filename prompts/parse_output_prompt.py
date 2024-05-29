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

## SEARCH RESULTS

## RECOMMENDATIONS 
"""

user_prompt = """
Please parse a markdown file with following texts:
fault: {fault_text}
query: {query_text}
answer: {answer_text}
rec: {rec_text}
"""