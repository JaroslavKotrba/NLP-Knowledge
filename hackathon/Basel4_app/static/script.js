document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('queryForm');
    const queryInput = document.getElementById('query');
    const replyElement = document.getElementById('reply');
    const categoryElement = document.getElementById('category');
    const confidenceElement = document.getElementById('confidence');
    const resultDiv = document.getElementById('result');
    const submitButton = form.querySelector('button[type="submit"]');
    // Store the original button content
    const originalButtonContent = submitButton.innerHTML;

    form.addEventListener('submit', async function (event) {
        event.preventDefault();

        // Disable the submit button and show loading state
        submitButton.disabled = true;
        submitButton.textContent = 'Processing...';

        // Clear previous results
        replyElement.textContent = '';
        categoryElement.textContent = '';
        confidenceElement.textContent = '';

        const query = queryInput.value;

        try {
            const response = await fetch('/ask/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: new URLSearchParams({ 'query': query })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            // Update the DOM with the response
            replyElement.textContent = result.reply;
            categoryElement.textContent = result.category;
            confidenceElement.textContent = result.confidence
                ? `${(result.confidence * 100).toFixed(2)}%`
                : 'N/A';

            // Show the result div
            resultDiv.style.display = 'block';
        } catch (error) {
            console.error('Error:', error);
            replyElement.textContent = 'An error occurred while processing your request.';
            categoryElement.textContent = 'Error';
            confidenceElement.textContent = 'N/A';
            resultDiv.style.display = 'block';
        } finally {
            // Re-enable the submit button and reset its text
            submitButton.disabled = false;
            submitButton.innerHTML = originalButtonContent;
        }
    });
});

// Function to auto-resize textarea
function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
}

// Add event listener to textarea
document.getElementById('query').addEventListener('input', function () {
    autoResize(this);
});
