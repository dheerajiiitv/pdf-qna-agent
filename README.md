# PDF QA System

This system leverages OpenAI's LLMs to answer questions based on PDF content and post results to Slack.

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in your credentials
4. Run the application: `uvicorn app.app:main_app --reload`

## Usage

Send a POST request to `/api/v1/qa` with:
- PDF file
- List of questions
- Command to process and post to Slack

Sample request body:
```
{
  "user_query": "Please answer the question and post it on slack",
  "request": {
    "pdf_file": "/Users/dheerajagrawal/Documents/QnAAgent/handbook.pdf",
      "questions": [
        {
          "text": "What is this document about"
        }
      ]
    
  }
}
```

Sample response:
```
"The question has been answered and the response has been posted on Slack. The document is an employee handbook for Zania, Inc., outlining company policies, benefits, ethics, and procedures for employees."
```
