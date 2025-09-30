import { Conversation } from '@elevenlabs/client';

const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const connectionStatus = document.getElementById('connectionStatus');
const agentStatus = document.getElementById('agentStatus');
const userAvatar = document.getElementById('userAvatar');
const agentVideo = document.getElementById('agentVideo');
const otterImage = document.getElementById('otterImage');

let conversation;
let isAgentSpeaking = false; // Flag to track if the agent is speaking

async function startConversation() {
    try {
        await navigator.mediaDevices.getUserMedia({ audio: true });

        conversation = await Conversation.startSession({
            agentId: 'your_agent_id_here',  //PLACE YOUR_AGENT_ID_HERE
            apiKey: 'your_api_key_here',    // YOUR_API_KEY_HERE
            onConnect: () => {
                connectionStatus.textContent = 'Connected';
                startButton.disabled = true;
                stopButton.disabled = false;
            },
            onDisconnect: () => {
                connectionStatus.textContent = 'Disconnected';
                startButton.disabled = false;
                stopButton.disabled = true;
                agentVideo.style.display = 'none';
                agentVideo.pause();
                otterImage.style.display = 'block';
            },
            onError: (error) => {
                console.error('Error:', error);
            },
            onModeChange: (mode) => {
                if (mode.mode === 'speaking') {
                    if (!isAgentSpeaking) {
                        isAgentSpeaking = true;
                        agentStatus.textContent = 'speaking';
                        otterImage.style.display = 'none';
                        agentVideo.style.display = 'block';
                        agentVideo.play();
                    } else {
                        console.warn('Agent is already speaking. Ignoring additional speech.');
                    }
                } else {
                    isAgentSpeaking = false;
                    agentStatus.textContent = 'listening';
                    agentVideo.style.display = 'none';
                    agentVideo.pause();
                    otterImage.style.display = 'block';
                }
            },
        });
    } catch (error) {
        console.error('Failed to start conversation:', error);
    }
}

async function stopConversation() {
    if (conversation) {
        await conversation.endSession();
        conversation = null;
    }
}

startButton.addEventListener('click', startConversation);
stopButton.addEventListener('click', stopConversation);
