// dashboard.js - Enhanced Dashboard functionality
class DashboardManager {
    constructor() {
        this.aiApiBase = 'http://localhost:5000/api';
        this.authApiBase = 'http://localhost:3000/api/auth';
        this.predictionHistory = JSON.parse(localStorage.getItem('agriverse-predictions')) || [];
        this.analyticsData = null;
        this.init();
    }

    init() {
        this.setupNavigation();
        this.setupEventListeners();
        this.updateStats();
        this.loadPredictionHistory();
        this.checkUserAuthentication();
        this.loadAnalytics();
    }

    checkUserAuthentication() {
        const token = localStorage.getItem('agriverse-token');
        const user = localStorage.getItem('agriverse-user');
        
        if (!token || !user) {
            setTimeout(() => {
                window.location.href = 'index.html';
            }, 1000);
            return;
        }
        
        try {
            const userData = JSON.parse(user);
            document.getElementById('userName').textContent = userData.firstName || 'User';
        } catch (e) {
            console.error('Error parsing user data:', e);
        }
    }

    setupNavigation() {
        const navLinks = document.querySelectorAll('.nav-link');
        const sections = document.querySelectorAll('.dashboard-section');

        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                
                // Remove active class from all links and sections
                navLinks.forEach(l => l.classList.remove('active'));
                sections.forEach(s => s.classList.remove('active'));
                
                // Add active class to clicked link
                link.classList.add('active');
                
                // Show corresponding section
                const targetId = link.getAttribute('href').substring(1);
                const targetSection = document.getElementById(targetId);
                if (targetSection) {
                    targetSection.classList.add('active');
                    targetSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            });
        });
    }

    setupEventListeners() {
        // Yield prediction form
        const yieldForm = document.getElementById('yieldForm');
        if (yieldForm) {
            yieldForm.addEventListener('submit', (e) => this.handleYieldPrediction(e));
        }

        // Disease detection
        const imageUpload = document.getElementById('imageUpload');
        if (imageUpload) {
            imageUpload.addEventListener('change', (e) => this.handleImageUpload(e));
        }

        const analyzeBtn = document.getElementById('analyzeBtn');
        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', () => this.analyzeDisease());
        }

        // User menu
        const userBtn = document.getElementById('userBtn');
        if (userBtn) {
            userBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                document.querySelector('.user-dropdown').classList.toggle('active');
            });
        }

        // Close dropdown when clicking outside
        document.addEventListener('click', () => {
            const dropdown = document.querySelector('.user-dropdown');
            if (dropdown) dropdown.classList.remove('active');
        });
    }

    async handleYieldPrediction(event) {
        event.preventDefault();
        
        const formData = {
            ndvi: parseFloat(document.getElementById('ndvi').value),
            rainfall: parseFloat(document.getElementById('rainfall').value),
            temperature: parseFloat(document.getElementById('temperature').value),
            humidity: parseFloat(document.getElementById('humidity').value),
            crop_type: document.getElementById('cropType').value,
            soil_ph: parseFloat(document.getElementById('soilPh').value),
            nitrogen_level: parseFloat(document.getElementById('nitrogenLevel').value),
            evi: parseFloat(document.getElementById('evi').value)
        };

        // Validate form data
        const validation = this.validateYieldForm(formData);
        if (!validation.isValid) {
            this.showError('Please correct the following errors: ' + Object.values(validation.errors).join(', '));
            return;
        }

        this.showLoading('Predicting crop yield with comprehensive analysis...');

        try {
            const response = await fetch(`${this.aiApiBase}/predict/yield`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('agriverse-token')}`
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();

            if (result.success) {
                this.displayYieldResults(result);
                this.savePrediction('yield', formData, result);
                this.updateStats();
                window.showNotification('Yield prediction completed successfully!', 'success');
            } else {
                this.showError('Yield prediction failed: ' + (result.error || 'Unknown error'));
            }
        } catch (error) {
            console.error('Yield prediction error:', error);
            this.showError('Failed to connect to prediction service. Please check your connection.');
        } finally {
            this.hideLoading();
        }
    }

    validateYieldForm(formData) {
        const errors = {};
        
        if (formData.ndvi < 0 || formData.ndvi > 1) {
            errors.ndvi = 'NDVI must be between 0 and 1';
        }
        
        if (formData.rainfall < 0) {
            errors.rainfall = 'Rainfall cannot be negative';
        }
        
        if (formData.temperature < -50 || formData.temperature > 60) {
            errors.temperature = 'Temperature must be between -50°C and 60°C';
        }
        
        if (formData.humidity < 0 || formData.humidity > 100) {
            errors.humidity = 'Humidity must be between 0% and 100%';
        }
        
        if (formData.soil_ph < 4 || formData.soil_ph > 9) {
            errors.soil_ph = 'Soil pH must be between 4 and 9';
        }
        
        if (formData.nitrogen_level < 0 || formData.nitrogen_level > 200) {
            errors.nitrogen_level = 'Nitrogen level must be between 0 and 200 ppm';
        }
        
        if (formData.evi < 0 || formData.evi > 1) {
            errors.evi = 'EVI must be between 0 and 1';
        }
        
        return {
            isValid: Object.keys(errors).length === 0,
            errors
        };
    }

    displayYieldResults(result) {
        // Update main results
        document.getElementById('predictedYield').textContent = 
            result.prediction.predicted_yield.toLocaleString();
        document.getElementById('confidenceLevel').textContent = 
            (result.prediction.confidence * 100).toFixed(1) + '%';
        document.getElementById('riskLevel').textContent = result.analytics.risk_level;
        document.getElementById('riskLevel').className = `risk-level ${result.analytics.risk_level.toLowerCase()}`;

        // Update yield range
        document.getElementById('yieldMin').textContent = 
            result.analytics.expected_yield_range.min.toLocaleString();
        document.getElementById('yieldMax').textContent = 
            result.analytics.expected_yield_range.max.toLocaleString();
        document.getElementById('yieldAverage').textContent = 
            result.analytics.expected_yield_range.average.toLocaleString();

        // Update range bar visualization
        const rangeFill = document.getElementById('yieldRangeFill');
        const min = result.analytics.expected_yield_range.min;
        const max = result.analytics.expected_yield_range.max;
        const current = result.prediction.predicted_yield;
        const percentage = ((current - min) / (max - min)) * 100;
        rangeFill.style.width = Math.max(10, Math.min(100, percentage)) + '%';

        // Update comparison data
        document.getElementById('yourYield').textContent = 
            result.prediction.predicted_yield.toLocaleString();
        document.getElementById('regionalAverage').textContent = 
            result.analytics.comparison_to_average.regional_average.toLocaleString();
        
        const difference = result.analytics.comparison_to_average.difference;
        const differenceElement = document.getElementById('yieldDifference');
        differenceElement.textContent = difference >= 0 ? `+${difference}` : difference;
        differenceElement.className = `difference-value ${difference >= 0 ? 'positive' : 'negative'}`;

        // Update factors analysis
        const factorsList = document.getElementById('factorsList');
        factorsList.innerHTML = '';
        if (result.analytics.factors_analysis.length > 0) {
            result.analytics.factors_analysis.forEach(factor => {
                const factorItem = document.createElement('div');
                factorItem.className = 'factor-item';
                factorItem.innerHTML = `
                    <i class="fas fa-exclamation-circle"></i>
                    <span>${factor}</span>
                `;
                factorsList.appendChild(factorItem);
            });
        } else {
            factorsList.innerHTML = '<div class="factor-item positive"><i class="fas fa-check-circle"></i><span>All factors appear favorable</span></div>';
        }

        // Update recommendations
        const immediateActions = document.getElementById('immediateActions');
        immediateActions.innerHTML = '';
        result.recommendations.immediate_actions.forEach(action => {
            const li = document.createElement('li');
            li.textContent = action;
            immediateActions.appendChild(li);
        });

        const longTermStrategies = document.getElementById('longTermStrategies');
        longTermStrategies.innerHTML = '';
        result.recommendations.long_term_strategies.forEach(strategy => {
            const li = document.createElement('li');
            li.textContent = strategy;
            longTermStrategies.appendChild(li);
        });

        // Update weather impact
        const weatherInfo = document.getElementById('weatherInfo');
        if (result.weather_impact.current_conditions.success) {
            weatherInfo.innerHTML = `
                <div class="weather-current">
                    <div class="weather-item">
                        <i class="fas fa-thermometer-half"></i>
                        <span>Temperature: ${result.weather_impact.current_conditions.temperature}°C</span>
                    </div>
                    <div class="weather-item">
                        <i class="fas fa-tint"></i>
                        <span>Humidity: ${result.weather_impact.current_conditions.humidity}%</span>
                    </div>
                    <div class="weather-item">
                        <i class="fas fa-cloud-rain"></i>
                        <span>Rainfall: ${result.weather_impact.current_conditions.rainfall}mm</span>
                    </div>
                </div>
                <div class="weather-advisory">
                    <i class="fas fa-info-circle"></i>
                    <span>${result.weather_impact.advisory}</span>
                </div>
            `;
        } else {
            weatherInfo.innerHTML = '<p>Weather data unavailable. Check local forecast for accurate impact assessment.</p>';
        }

        // Show results section
        document.getElementById('yieldResults').classList.remove('hidden');
        
        // Scroll to results
        document.getElementById('yieldResults').scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
    }

    handleImageUpload(event) {
        const file = event.target.files[0];
        if (file) {
            // Validate file type
            const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
            if (!validTypes.includes(file.type)) {
                this.showError('Please upload JPG, PNG, or WebP images only.');
                event.target.value = '';
                return;
            }

            // Validate file size (5MB max)
            if (file.size > 5 * 1024 * 1024) {
                this.showError('Image size must be less than 5MB.');
                event.target.value = '';
                return;
            }

            this.previewImage(file);
            document.getElementById('analyzeBtn').classList.remove('hidden');
        }
    }

    previewImage(file) {
        const reader = new FileReader();
        const preview = document.getElementById('previewImage');
        const previewContainer = document.getElementById('imagePreview');
        const uploadArea = document.getElementById('uploadArea');

        reader.onload = function(e) {
            // Validate image dimensions
            const img = new Image();
            img.onload = function() {
                if (img.width < 100 || img.height < 100) {
                    window.showNotification('Image is too small. Please upload an image at least 100x100 pixels.', 'warning');
                    document.getElementById('imageUpload').value = '';
                    return;
                }
                
                preview.src = e.target.result;
                previewContainer.classList.remove('hidden');
                uploadArea.classList.add('hidden');
            };
            img.src = e.target.result;
        };

        reader.readAsDataURL(file);
    }

    removeImage() {
        document.getElementById('imageUpload').value = '';
        document.getElementById('imagePreview').classList.add('hidden');
        document.getElementById('uploadArea').classList.remove('hidden');
        document.getElementById('analyzeBtn').classList.add('hidden');
        document.getElementById('diseaseResults').classList.add('hidden');
    }

    async analyzeDisease() {
        const file = document.getElementById('imageUpload').files[0];
        if (!file) {
            this.showError('Please select an image first.');
            return;
        }

        this.showLoading('Analyzing plant disease with comprehensive assessment...');

        // Show analyzing status
        const analyzeBtn = document.getElementById('analyzeBtn');
        const originalText = analyzeBtn.innerHTML;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        analyzeBtn.disabled = true;

        try {
            const formData = new FormData();
            formData.append('image', file);

            const response = await fetch(`${this.aiApiBase}/predict/disease`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.displayDiseaseResults(result);
                this.savePrediction('disease', { filename: file.name }, result);
                this.updateStats();
                window.showNotification('Disease analysis completed successfully!', 'success');
            } else {
                this.handleDiseaseError(result);
            }
        } catch (error) {
            console.error('Disease detection error:', error);
            this.showError('Failed to connect to analysis service. Please check your connection.');
        } finally {
            this.hideLoading();
            analyzeBtn.innerHTML = originalText;
            analyzeBtn.disabled = false;
        }
    }

    handleDiseaseError(result) {
        switch (result.error_type) {
            case 'invalid_image':
                this.showError(result.error);
                this.removeImage();
                break;
            case 'low_confidence':
                if (confirm('⚠️ ' + result.error + ` (Confidence: ${(result.confidence * 100).toFixed(1)}%)\n\nDo you want to see the result anyway?`)) {
                    // Process with low confidence warning
                    this.displayDiseaseResults(result, true);
                }
                break;
            case 'uncertain_healthy':
                this.showError(result.error);
                break;
            default:
                this.showError('Analysis failed: ' + result.error);
        }
    }

    displayDiseaseResults(result, isLowConfidence = false) {
        const diseaseName = this.formatDiseaseName(result.prediction_details.disease_name);
        
        // Update main disease info
        document.getElementById('diseaseName').textContent = isLowConfidence ? 
            `[Low Confidence] ${diseaseName}` : diseaseName;
        
        document.getElementById('confidenceValue').textContent = 
            (result.prediction_details.confidence * 100).toFixed(1);
        
        // Update severity badge
        const severityBadge = document.getElementById('severityBadge');
        severityBadge.textContent = result.disease_information.severity;
        severityBadge.className = `severity-badge ${result.disease_information.severity.toLowerCase()}`;

        // Update health status
        const healthStatus = document.getElementById('healthStatus');
        if (result.prediction_details.is_healthy) {
            healthStatus.innerHTML = '<span class="health-status healthy"><i class="fas fa-check-circle"></i> Healthy Plant</span>';
        } else {
            healthStatus.innerHTML = '<span class="health-status diseased"><i class="fas fa-exclamation-triangle"></i> Disease Detected</span>';
        }

        // Update disease description
        document.getElementById('diseaseDescription').textContent = result.disease_information.description;

        // Update alternative predictions
        const alternativePredictions = document.getElementById('alternativePredictions');
        alternativePredictions.innerHTML = '';
        result.alternative_predictions.forEach(pred => {
            const predItem = document.createElement('div');
            predItem.className = 'prediction-item';
            predItem.innerHTML = `
                <span class="prediction-name">${this.formatDiseaseName(pred.disease)}</span>
                <span class="prediction-confidence">${(pred.confidence * 100).toFixed(1)}%</span>
            `;
            alternativePredictions.appendChild(predItem);
        });

        // Update treatment recommendations
        const treatmentList = document.getElementById('treatmentList');
        treatmentList.innerHTML = '';
        result.disease_information.recommendations.forEach(treatment => {
            const li = document.createElement('li');
            li.textContent = treatment;
            treatmentList.appendChild(li);
        });

        // Update prevention measures
        const preventionList = document.getElementById('preventionList');
        preventionList.innerHTML = '';
        result.disease_information.prevention_measures.forEach(prevention => {
            const li = document.createElement('li');
            li.textContent = prevention;
            preventionList.appendChild(li);
        });

        // Update action plan
        const immediateActions = document.getElementById('immediateDiseaseActions');
        immediateActions.innerHTML = '';
        result.action_plan.immediate_actions.forEach(action => {
            const li = document.createElement('li');
            li.textContent = action;
            immediateActions.appendChild(li);
        });

        const monitoringAdvice = document.getElementById('monitoringAdvice');
        monitoringAdvice.innerHTML = '';
        result.action_plan.monitoring_advice.forEach(advice => {
            const li = document.createElement('li');
            li.textContent = advice;
            monitoringAdvice.appendChild(li);
        });

        document.getElementById('expertConsultation').textContent = result.action_plan.expert_consultation;

        // Show results section
        document.getElementById('diseaseResults').classList.remove('hidden');
        
        // Scroll to results
        document.getElementById('diseaseResults').scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
    }

    formatDiseaseName(disease) {
        return disease.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    async loadAnalytics() {
        try {
            const response = await fetch(`${this.aiApiBase}/analytics/history`);
            const result = await response.json();
            
            if (result.success) {
                this.analyticsData = result.analytics;
                this.updateAnalyticsDisplay();
            }
        } catch (error) {
            console.error('Analytics load error:', error);
        }
    }

    updateAnalyticsDisplay() {
        if (!this.analyticsData) return;

        // Update overview stats
        document.getElementById('totalPredictionsCount').textContent = this.analyticsData.total_predictions;
        document.getElementById('diseasePredictionsCount').textContent = 
            this.analyticsData.common_diseases.reduce((sum, disease) => sum + disease.count, 0);
        document.getElementById('yieldPredictionsCount').textContent = this.analyticsData.yield_predictions.length;
        
        const avgAccuracy = this.analyticsData.accuracy_trend.reduce((a, b) => a + b, 0) / this.analyticsData.accuracy_trend.length;
        document.getElementById('averageAccuracy').textContent = (avgAccuracy * 100).toFixed(1) + '%';

        // Update common diseases
        const commonDiseasesList = document.getElementById('commonDiseasesList');
        commonDiseasesList.innerHTML = '';
        this.analyticsData.common_diseases.forEach(disease => {
            const diseaseItem = document.createElement('div');
            diseaseItem.className = 'disease-item';
            diseaseItem.innerHTML = `
                <span class="disease-name">${this.formatDiseaseName(disease.disease)}</span>
                <span class="disease-count">${disease.count} detections</span>
            `;
            commonDiseasesList.appendChild(diseaseItem);
        });

        // Update recent predictions
        this.updateRecentPredictions();
    }

    updateRecentPredictions() {
        const recentPredictions = document.getElementById('recentPredictions');
        recentPredictions.innerHTML = '';
        
        // Show last 5 predictions from history
        const recent = this.predictionHistory.slice(0, 5);
        
        if (recent.length === 0) {
            recentPredictions.innerHTML = '<p class="no-predictions">No predictions yet. Start analyzing!</p>';
            return;
        }

        recent.forEach(prediction => {
            const predictionItem = document.createElement('div');
            predictionItem.className = 'prediction-history-item';
            
            const date = new Date(prediction.timestamp).toLocaleDateString();
            let content = '';
            
            if (prediction.type === 'yield') {
                content = `
                    <i class="fas fa-seedling"></i>
                    <div class="prediction-info">
                        <span class="prediction-type">Yield Prediction</span>
                        <span class="prediction-details">${prediction.result.prediction.predicted_yield} kg/ha</span>
                    </div>
                    <span class="prediction-date">${date}</span>
                `;
            } else {
                content = `
                    <i class="fas fa-leaf"></i>
                    <div class="prediction-info">
                        <span class="prediction-type">Disease Detection</span>
                        <span class="prediction-details">${this.formatDiseaseName(prediction.result.prediction_details.disease_name)}</span>
                    </div>
                    <span class="prediction-date">${date}</span>
                `;
            }
            
            predictionItem.innerHTML = content;
            recentPredictions.appendChild(predictionItem);
        });
    }

    savePrediction(type, input, result) {
        const prediction = {
            id: Date.now(),
            type: type,
            timestamp: new Date().toISOString(),
            input: input,
            result: result,
            user: JSON.parse(localStorage.getItem('agriverse-user'))?.email
        };

        this.predictionHistory.unshift(prediction);
        
        // Keep only last 50 predictions
        if (this.predictionHistory.length > 50) {
            this.predictionHistory = this.predictionHistory.slice(0, 50);
        }

        localStorage.setItem('agriverse-predictions', JSON.stringify(this.predictionHistory));
        this.updateStats();
        this.updateRecentPredictions();
    }

    updateStats() {
        const totalPredictions = document.getElementById('totalPredictions');
        if (totalPredictions) {
            totalPredictions.textContent = this.predictionHistory.length;
        }

        // Calculate diseases detected
        const diseaseCount = this.predictionHistory.filter(p => 
            p.type === 'disease' && !p.result.prediction_details.is_healthy
        ).length;
        
        const diseasesDetected = document.getElementById('diseasesDetected');
        if (diseasesDetected) {
            diseasesDetected.textContent = diseaseCount;
        }
    }

    showLoading(message = 'Processing your request...') {
        const loading = document.getElementById('loading');
        if (loading) {
            const messageEl = loading.querySelector('p');
            if (messageEl) {
                messageEl.textContent = message;
            }
            loading.classList.remove('hidden');
        }
    }

    hideLoading() {
        const loading = document.getElementById('loading');
        if (loading) loading.classList.add('hidden');
    }

    showError(message) {
        window.showNotification(message, 'error', 5000);
    }

    logout() {
        localStorage.removeItem('agriverse-token');
        localStorage.removeItem('agriverse-user');
        localStorage.removeItem('agriverse-predictions');
        window.showNotification('Logged out successfully', 'info', 2000);
        
        setTimeout(() => {
            window.location.href = 'index.html';
        }, 1000);
    }
}

// Utility functions
function showProfile() {
    window.showNotification('Profile settings will be available in the next update', 'info');
}

function showPredictionHistory() {
    window.showNotification('Full prediction history will be available in the next update', 'info');
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Check if user is authenticated
    const token = localStorage.getItem('agriverse-token');
    if (!token) {
        window.location.href = 'index.html';
        return;
    }

    window.dashboardManager = new DashboardManager();
    
    // Make removeImage function globally available
    window.removeImage = () => window.dashboardManager.removeImage();
    window.logout = () => window.dashboardManager.logout();
});