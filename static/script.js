// Set up WebSocket connection
const ws = new WebSocket("ws://localhost:8009/ws");

let currentMessage = ""; // Stores the message being built word by word
let typingElement = null; // Tracks the "typing" element to prevent duplication
const converter = new showdown.Converter(); // Initialize Markdown converter for RAG responses

// WebSocket message handling
ws.onmessage = async function (event) {
    const message = event.data;

    if (message === "[END]") {
        // Convert the final Markdown message to HTML
        const messageHtml = converter.makeHtml(currentMessage);

        // Display the final assistant response
        displayAssistantMessage(messageHtml, true);

        // Reset the current message
        currentMessage = "";
    } else if (message === "NO_RELEVANT_DOCUMENT") {
        console.log("No relevant documents found. Falling back to handleFTQuery.");
        const input = document.getElementById("message-input");
        const fallbackResponse = await handleFTQueryFallback(input.value.trim());
        displayAssistantMessage(converter.makeHtml(fallbackResponse), true);
        currentMessage = ""; // Reset after fallback
    } else {
        // Add the new word to the current message progressively
        currentMessage += (currentMessage ? " " : "") + message;

        // Show the typing indicator with the current progressive message
        displayAssistantMessage(currentMessage, false);
    }
};

// Function to send user messages via WebSocket
function sendMessage() {
    const input = document.getElementById("message-input");
    const userMessage = input.value.trim();

    if (userMessage) {
        // Send the user's message to the WebSocket server
        ws.send(userMessage);

        // Display the user's message in the chat
        displayMessage("User", userMessage);

        // Clear the input field
        input.value = "";
    }
}

// Function to display the assistant's message (progressively or final)
function displayAssistantMessage(content, isFinal) {
    const messages = document.getElementById("messages");
    if (!typingElement) {
        typingElement = document.createElement("li");
        typingElement.className = "assistant";
        typingElement.id = "typing";
        messages.appendChild(typingElement);
    }

    // Update the content of the typing element for progressive display
    if (!isFinal) {
        typingElement.innerHTML = content;
    } else {
        // Create a new element for the finalized message
        const finalMessageElement = document.createElement("li");
        finalMessageElement.className = "assistant";
        finalMessageElement.innerHTML = content; // Render Markdown for final message
        messages.appendChild(finalMessageElement);

        // Remove the typing indicator
        removeTypingIndicator();
    }

    // Scroll to the bottom of the chat container
    messages.scrollTop = messages.scrollHeight;
}

// Function to remove the typing indicator
function removeTypingIndicator() {
    if (typingElement) {
        typingElement.remove();
        typingElement = null;
    }
}

// Unified API fallback for non-document-related queries
async function handleFTQueryFallback(query) {
    try {
        const response = await fetch("/ft_rag/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: query, top_k: 0 }) // Set top_k=0 for non-retrieval queries
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }

        const data = await response.json();
        return data.response; // Ensure the API returns a `response` field
    } catch (error) {
        console.error("Error fetching fallback FT-RAG response:", error);
        return "An error occurred while processing your query.";
    }
}

// Function to display messages in the chat
function displayMessage(sender, content) {
    const messages = document.getElementById("messages");
    if (!messages) {
        console.error("Chat container not found. Ensure an element with ID 'messages' exists.");
        return;
    }

    // Create a new list item for the message
    const messageElement = document.createElement("li");
    messageElement.className = sender.toLowerCase(); // Assign class based on sender ('user' or 'assistant')

    // Add sender's name and content
    messageElement.innerHTML = `${content}`;

    // Append the message to the chat container
    messages.appendChild(messageElement);

    // Scroll the chat container to the bottom
    messages.scrollTop = messages.scrollHeight;
}
