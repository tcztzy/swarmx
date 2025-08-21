# Testing with Real OpenAI API

The SwarmX test suite supports running tests against the real OpenAI API using the `--openai` flag.

## Usage

```bash
# Run all tests with real OpenAI API calls
pytest --openai

# Run specific test with real OpenAI API calls
pytest --openai tests/test_agent.py::test_create_chat_completion

# Use a specific model (default is gpt-4o-mini when --openai is used)
pytest --openai --model gpt-4o
```

## Setup

To run tests with real OpenAI API calls, you need to provide a valid OpenAI API key:

### Option 1: Environment Variable (Recommended)
```bash
export REAL_OPENAI_API_KEY="your-actual-openai-api-key"
pytest --openai
```

### Option 2: Set in your shell session
```bash
REAL_OPENAI_API_KEY="your-actual-openai-api-key" pytest --openai
```

## What happens when using `--openai`

1. **Mocking is disabled**: Tests make real API calls instead of using mock responses
2. **Model selection**: If using the default model (`deepseek-reasoner`), it's automatically changed to `gpt-4o-mini` for compatibility with OpenAI API
3. **API endpoint**: Forces the use of `https://api.openai.com/v1` regardless of local `.env` settings
4. **API key**: Uses `REAL_OPENAI_API_KEY` environment variable if available, otherwise uses a placeholder (which will cause authentication errors)

## Expected behavior

- **With valid API key**: Tests should pass and make real API calls
- **Without valid API key**: Tests will fail with authentication errors (401), but this confirms the real API is being called
- **With invalid model**: Tests will fail with model not found errors (404)

## Cost considerations

⚠️ **Warning**: Running tests with `--openai` will make real API calls and consume your OpenAI API credits. Use with caution, especially when running the full test suite.

## Troubleshooting

### Authentication Error (401)
```
openai.AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided'...
```
**Solution**: Set the `REAL_OPENAI_API_KEY` environment variable with a valid OpenAI API key.

### Model Not Found (404)
```
openai.NotFoundError: Error code: 404 - {'error': {'message': 'model "..." not found'...
```
**Solution**: Use a valid OpenAI model with the `--model` option:
```bash
pytest --openai --model gpt-4o-mini
```

### Still using local server
If tests are still hitting a local server (like Ollama), check that:
1. The test fixture is properly overriding environment variables
2. You're using the `--openai` flag
3. The test is using the `model` fixture parameter

## Implementation details

The `--openai` option works by:
1. Setting `is_mocking` fixture to `False`
2. Overriding `OPENAI_BASE_URL` to `https://api.openai.com/v1`
3. Using `REAL_OPENAI_API_KEY` if available, otherwise a placeholder
4. Automatically selecting `gpt-4o-mini` model when the default `deepseek-reasoner` is used
5. Skipping all mock setup and patching
