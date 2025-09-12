const API_URL = 'http://localhost:8000';
        let isOffline = false;
        let offlineData = {};
        
        // Check if API is available
        async function checkConnection() {
            try {
                const response = await fetch(API_URL + '/', { method: 'GET' });
                if (!response.ok) throw new Error('API not available');
                isOffline = false;
                document.body.classList.remove('offline');
            } catch (error) {
                isOffline = true;
                document.body.classList.add('offline');
                loadOfflineData();
            }
        }
        
        // Load offline data
        async function loadOfflineData() {
            try {
                const response = await fetch(API_URL + '/offline-data');
                offlineData = await response.json();
                localStorage.setItem('offlineData', JSON.stringify(offlineData));
            } catch (error) {
                // Try to load from localStorage
                const stored = localStorage.getItem('offlineData');
                if (stored) {
                    offlineData = JSON.parse(stored);
                }
            }
        }
        
        // Switch tabs
        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            if (tab === 'location') {
                document.querySelector('.tabs .tab:first-child').classList.add('active');
                document.getElementById('location-tab').classList.add('active');
            } else {
                document.querySelector('.tabs .tab:last-child').classList.add('active');
                document.getElementById('manual-tab').classList.add('active');
            }
        }
        
        // Handle location form submission
        document.getElementById('location-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = {
                location: {
                    latitude: parseFloat(formData.get('latitude')),
                    longitude: parseFloat(formData.get('longitude')),
                    region: formData.get('region')
                },
                language: formData.get('language')
            };
            
            await getRecommendations(data);
        });
        
        // Handle manual form submission
        document.getElementById('manual-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const soilData = {};
            
            for (let [key, value] of formData.entries()) {
                if (value) {
                    soilData[key] = parseFloat(value);
                }
            }
            
            const data = {
                location: {
                    latitude: 23.3441,
                    longitude: 85.3096,
                    region: "Jharkhand"
                },
                soil_data: soilData,
                language: document.getElementById('language').value
            };
            
            await getRecommendations(data);
        });
        
        // Get recommendations from API
        async function getRecommendations(data) {
            const recommendationsDiv = document.getElementById('recommendations');
            recommendationsDiv.innerHTML = '<div class="loading"><div class="spinner"></div>Getting recommendations...</div>';
            
            try {
                if (isOffline) {
                    // Use offline recommendations
                    displayOfflineRecommendations(data);
                    return;
                }
                
                const response = await fetch(API_URL + '/recommend', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    displayRecommendations(result.recommendations, result.environmental_data);
                } else {
                    recommendationsDiv.innerHTML = '<p style="color: red;">Error getting recommendations</p>';
                }
            } catch (error) {
                console.error('Error:', error);
                recommendationsDiv.innerHTML = '<p style="color: red;">Failed to connect to server</p>';
            }
        }
        
        // Display recommendations
        function displayRecommendations(recommendations, environmentalData) {
            const recommendationsDiv = document.getElementById('recommendations');
            let html = '<h3>Recommended Crops</h3>';
            
            if (environmentalData) {
                html += `
                    <div style="margin-bottom: 15px; padding: 10px; background: #f7fafc; border-radius: 5px;">
                        <strong>Environmental Conditions:</strong><br>
                        Soil pH: ${environmentalData.soil.ph}<br>
                        Temperature: ${environmentalData.weather.temperature}°C<br>
                        Humidity: ${environmentalData.weather.humidity}%
                    </div>
                `;
            }
            
            recommendations.forEach(rec => {
                html += `
                    <div class="recommendation-card">
                        <div class="crop-name">${rec.crop}</div>
                        <div class="crop-details">
                            <div class="detail-item">
                                <span class="detail-label">Confidence:</span> ${(rec.confidence * 100).toFixed(1)}%
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: ${rec.confidence * 100}%"></div>
                                </div>
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">Expected Yield:</span> ${rec.predicted_yield} tons/ha
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">Profit Margin:</span> ${rec.profit_margin}
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">Est. Profit:</span> ₹${rec.estimated_profit}/ha
                            </div>
                        </div>
                        ${rec.market_info ? `
                            <div class="detail-item">
                                <span class="detail-label">Market Price:</span> ₹${rec.market_info.current_price}/quintal
                                (${rec.market_info.price_trend} trend, ${rec.market_info.demand} demand)
                            </div>
                        ` : ''}
                        <div class="care-tips">
                            <h4>Care Tips:</h4>
                            <ul>
                                ${rec.care_tips.map(tip => `<li>${tip}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                `;
            });
            
            recommendationsDiv.innerHTML = html;
        }
        
        // Display offline recommendations
        function displayOfflineRecommendations(data) {
            const recommendationsDiv = document.getElementById('recommendations');
            
            if (!offlineData.data || !offlineData.data.crop_database) {
                recommendationsDiv.innerHTML = '<p>Offline data not available</p>';
                return;
            }
            
            // Simple offline recommendation based on stored crop database
            const crops = Object.keys(offlineData.data.crop_database);
            const recommendations = crops.slice(0, 3).map(crop => {
                const cropInfo = offlineData.data.crop_database[crop];
                return {
                    crop: crop,
                    confidence: Math.random() * 0.3 + 0.6,
                    predicted_yield: Math.random() * 3 + 3,
                    estimated_profit: Math.random() * 50000 + 50000,
                    profit_margin: cropInfo.profit_margin,
                    sustainability_score: cropInfo.sustainability_score,
                    suitable_season: cropInfo.season,
                    care_tips: ["Consult local agricultural expert", "Monitor crop regularly"]
                };
            });
            
            displayRecommendations(recommendations, null);
        }
        
        // Send chat message
        async function sendMessage() {
            const input = document.getElementById('chat-input');
            const message = input.value.trim();
            
            if (!message) return;
            
            const messagesDiv = document.getElementById('chat-messages');
            
            // Add user message
            messagesDiv.innerHTML += `<div class="message user-message">${message}</div>`;
            input.value = '';
            
            // Show loading
            messagesDiv.innerHTML += `<div class="message bot-message loading">Thinking...</div>`;
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            try {
                if (isOffline) {
                    // Use offline responses
                    setTimeout(() => {
                        const loadingMsg = messagesDiv.querySelector('.loading');
                        loadingMsg.remove();
                        
                        const response = getOfflineResponse(message);
                        messagesDiv.innerHTML += `<div class="message bot-message">${response}</div>`;
                        messagesDiv.scrollTop = messagesDiv.scrollHeight;
                    }, 500);
                    return;
                }
                
                const response = await fetch(API_URL + '/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: message,
                        language: document.getElementById('language').value
                    })
                });
                
                const result = await response.json();
                
                // Remove loading message
                const loadingMsg = messagesDiv.querySelector('.loading');
                loadingMsg.remove();
                
                if (result.status === 'success') {
                    const botResponse = formatChatResponse(result.response);
                    messagesDiv.innerHTML += `<div class="message bot-message">${botResponse}</div>`;
                } else {
                    messagesDiv.innerHTML += `<div class="message bot-message">Sorry, I couldn't process your request.</div>`;
                }
            } catch (error) {
                console.error('Error:', error);
                const loadingMsg = messagesDiv.querySelector('.loading');
                if (loadingMsg) loadingMsg.remove();
                messagesDiv.innerHTML += `<div class="message bot-message">Sorry, I'm having trouble connecting to the server.</div>`;
            }
            
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        // Format chat response
        function formatChatResponse(response) {
            let html = response.message || '';
            
            if (response.suggestions && response.suggestions.length > 0) {
                html += '<br><br><strong>Suggestions:</strong><ul>';
                response.suggestions.forEach(suggestion => {
                    html += `<li>${suggestion}</li>`;
                });
                html += '</ul>';
            }
            
            if (response.current_weather) {
                html += '<br><strong>Current Weather:</strong><br>';
                html += `Temperature: ${response.current_weather.temperature}<br>`;
                html += `Humidity: ${response.current_weather.humidity}<br>`;
                html += `Rainfall: ${response.current_weather.rainfall}`;
            }
            
            if (response.current_prices) {
                html += '<br><strong>Market Prices:</strong><br>';
                for (let crop in response.current_prices) {
                    html += `${crop}: ${response.current_prices[crop].price}<br>`;
                }
            }
            
            return html;
        }
        
        // Get offline response
        function getOfflineResponse(message) {
            const lowerMessage = message.toLowerCase();
            
            if (lowerMessage.includes('hello') || lowerMessage.includes('hi')) {
                return "Hello! I'm currently working offline, but I can still help with basic agricultural questions.";
            }
            
            if (lowerMessage.includes('crop') || lowerMessage.includes('recommend')) {
                return "For crop recommendations, please use the form on the left. In offline mode, I'll provide general recommendations based on cached data.";
            }
            
            if (lowerMessage.includes('weather')) {
                return "Weather data is not available in offline mode. Please check your local weather service.";
            }
            
            if (lowerMessage.includes('price') || lowerMessage.includes('market')) {
                return "Market prices are not available in offline mode. Please check with your local mandi when you're back online.";
            }
            
            return "I'm working in offline mode with limited capabilities. Please try again when you're connected to the internet for full functionality.";
        }
        
        // Allow Enter key to send message
        document.getElementById('chat-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Check connection on load
        checkConnection();
        setInterval(checkConnection, 30000); // Check every 30 seconds