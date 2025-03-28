GA1_PROMPTS = {
    "code_s": """Question: What is the output of running 'code -s' in the terminal?
The command 'code -s' is used to print VS Code's process ID and debugging information.
Provide the exact output that appears when VS Code is not found on the PATH.""",
    
    "httpbin": """Question: What is the JSON output when sending a HTTPS request to httpbin.org/get?
Format the response as a proper JSON object with the exact structure returned by httpbin.org.
Include only the response body, not headers or other information.""",
    
    "excel_formula": """Question: Given a spreadsheet question, provide the exact Excel/Google Sheets formula.
Format the answer as a valid spreadsheet formula starting with '='."""
}

GA2_PROMPTS = {
    "github": """Question: Process GitHub API related questions.
Provide only the relevant information from the GitHub API response.
Format as valid JSON when appropriate.""",
    
    "docker": """Question: Handle Docker-related questions.
Provide exact Docker commands or expected output.
For image pushes, use the format: 'username/repository:tag'"""
}

GA3_PROMPTS = {
    "colab": """Question: Process Google Colab related questions.
Provide realistic output that would be seen in a Colab notebook.
Include metrics for ML tasks (accuracy, loss) when relevant."""
} 