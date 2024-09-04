document.getElementById('send-btn').addEventListener('click', sendMessage);

function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    if (userInput.trim() === '') return;

    appendMessage('user-message', userInput);
    document.getElementById('user-input').value = '';

    fetch('/chatbot', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userInput }),
    })
    .then(response => response.json())
    .then(data => {
        appendMessage('bot-message', data.reply);
    })
    .catch(error => {
        console.error('Error:', error);
        appendMessage('bot-message', 'Oops! Something went wrong.');
    });
}

function appendMessage(className, message) {
    const output = document.getElementById('output');
    const messageElement = document.createElement('div');
    messageElement.className = `message ${className}`;
    messageElement.textContent = message;
    output.appendChild(messageElement);
    output.scrollTop = output.scrollHeight;
}
