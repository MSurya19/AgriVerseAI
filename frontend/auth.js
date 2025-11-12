// auth.js - Enhanced Authentication System for AgriVerseAI
class AuthManager {
    constructor() {
        this.apiBase = 'http://localhost:3000/api/auth';
        this.currentUser = null;
        this.init();
    }

    init() {
        this.checkAuthStatus();
        this.setupEventListeners();
    }

    // Check if user is already logged in
    async checkAuthStatus() {
        const token = localStorage.getItem('agriverse-token');
        if (token) {
            try {
                const response = await fetch(`${this.apiBase}/me`, {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                });
                
                if (response.ok) {
                    const data = await response.json();
                    this.currentUser = data.user;
                    localStorage.setItem('agriverse-user', JSON.stringify(data.user));
                    this.updateUIForAuth(data.user);
                    
                    // Redirect to dashboard if on landing page
                    if (window.location.pathname.endsWith('index.html') || window.location.pathname === '/') {
                        setTimeout(() => {
                            window.location.href = 'dashboard.html';
                        }, 1000);
                    }
                } else {
                    this.logout();
                }
            } catch (error) {
                console.error('Auth check failed:', error);
                this.logout();
            }
        } else if (window.location.pathname.includes('dashboard.html')) {
            // Redirect to login if trying to access dashboard without auth
            window.location.href = 'index.html';
        }
    }

    // Handle user login
    async handleLogin(event) {
        event.preventDefault();
        
        const email = document.getElementById('loginEmail').value;
        const password = document.getElementById('loginPassword').value;

        this.showLoading();

        try {
            const response = await fetch(`${this.apiBase}/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ email, password })
            });

            const data = await response.json();

            if (response.ok) {
                localStorage.setItem('agriverse-token', data.token);
                localStorage.setItem('agriverse-user', JSON.stringify(data.user));
                this.currentUser = data.user;
                this.showMessage('Login successful! Redirecting...', 'success');
                
                setTimeout(() => {
                    window.location.href = 'dashboard.html';
                }, 1500);
            } else {
                this.showMessage(data.message || 'Login failed', 'error');
            }
        } catch (error) {
            console.error('Login error:', error);
            this.showMessage('Network error. Please try again.', 'error');
        } finally {
            this.hideLoading();
        }
    }

    // Handle user registration
    async handleSignup(event) {
        event.preventDefault();
        
        const firstName = document.getElementById('signupFirstName').value;
        const lastName = document.getElementById('signupLastName').value;
        const email = document.getElementById('signupEmail').value;
        const password = document.getElementById('signupPassword').value;
        const confirmPassword = document.getElementById('signupConfirmPassword').value;

        if (password !== confirmPassword) {
            this.showMessage('Passwords do not match', 'error');
            return;
        }

        if (password.length < 6) {
            this.showMessage('Password must be at least 6 characters long', 'error');
            return;
        }

        this.showLoading();

        try {
            const response = await fetch(`${this.apiBase}/signup`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    firstName,
                    lastName,
                    email,
                    password
                })
            });

            const data = await response.json();

            if (response.ok) {
                this.showMessage('Account created successfully! Welcome to AgriVerseAI.', 'success');
                setTimeout(() => {
                    this.switchAuthForm('login');
                    this.clearAuthForm('signup');
                }, 2000);
            } else {
                this.showMessage(data.message || 'Registration failed', 'error');
            }
        } catch (error) {
            console.error('Signup error:', error);
            this.showMessage('Network error. Please try again.', 'error');
        } finally {
            this.hideLoading();
        }
    }

    // Handle forgot password
    async handleForgotPassword(event) {
        event.preventDefault();
        
        const email = document.getElementById('forgotEmail').value;

        this.showLoading();

        try {
            const response = await fetch(`${this.apiBase}/forgot-password`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ email })
            });

            const data = await response.json();

            if (response.ok) {
                this.showMessage('Password reset instructions sent to your email.', 'success');
                this.clearAuthForm('forgot');
            } else {
                this.showMessage(data.message || 'Failed to send reset email', 'error');
            }
        } catch (error) {
            console.error('Forgot password error:', error);
            this.showMessage('Network error. Please try again.', 'error');
        } finally {
            this.hideLoading();
        }
    }

    // Logout user
    logout() {
        localStorage.removeItem('agriverse-token');
        localStorage.removeItem('agriverse-user');
        localStorage.removeItem('agriverse-predictions');
        this.currentUser = null;
        window.location.href = 'index.html';
    }

    // Update UI based on auth status
    updateUIForAuth(user) {
        const userNameElement = document.getElementById('userName');
        if (userNameElement) {
            userNameElement.textContent = user.firstName;
        }
    }

    // Utility methods
    showMessage(message, type) {
        // Remove existing messages
        const existingMessage = document.querySelector('.auth-message');
        if (existingMessage) {
            existingMessage.remove();
        }

        const messageEl = document.createElement('div');
        messageEl.className = `auth-message ${type}`;
        messageEl.textContent = message;

        const activeForm = document.querySelector('.auth-form.active');
        if (activeForm) {
            const formContent = activeForm.querySelector('.auth-form-content');
            if (formContent) {
                formContent.prepend(messageEl);
            }
        }

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (messageEl.parentNode) {
                messageEl.remove();
            }
        }, 5000);
    }

    showLoading() {
        const loading = document.getElementById('loading');
        if (loading) loading.classList.remove('hidden');
    }

    hideLoading() {
        const loading = document.getElementById('loading');
        if (loading) loading.classList.add('hidden');
    }

    clearAuthForm(formType) {
        const forms = {
            'login': ['loginEmail', 'loginPassword'],
            'signup': ['signupFirstName', 'signupLastName', 'signupEmail', 'signupPassword', 'signupConfirmPassword'],
            'forgot': ['forgotEmail']
        };

        if (forms[formType]) {
            forms[formType].forEach(id => {
                const element = document.getElementById(id);
                if (element) element.value = '';
            });
        }
    }

    setupEventListeners() {
        // Password toggle functionality
        document.addEventListener('click', (e) => {
            if (e.target.closest('.password-toggle')) {
                const toggle = e.target.closest('.password-toggle');
                const input = toggle.closest('.input-group').querySelector('input');
                const icon = toggle.querySelector('i');
                
                if (input.type === 'password') {
                    input.type = 'text';
                    icon.className = 'fas fa-eye-slash';
                } else {
                    input.type = 'password';
                    icon.className = 'fas fa-eye';
                }
            }
        });

        // Close modal when clicking outside
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal-backdrop')) {
                this.closeAuthModal();
            }
        });

        // Escape key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeAuthModal();
            }
        });
    }
}

// Modal control functions
function openAuthModal(formType = 'login') {
    const modal = document.getElementById('authModal');
    if (modal) {
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
        switchAuthForm(formType);
    }
}

function closeAuthModal() {
    const modal = document.getElementById('authModal');
    if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = '';
        
        // Clear all forms
        ['login', 'signup', 'forgot'].forEach(form => {
            authManager.clearAuthForm(form);
        });
    }
}

function switchAuthForm(formType) {
    // Hide all forms
    document.querySelectorAll('.auth-form').forEach(form => {
        form.classList.remove('active');
    });
    
    // Show selected form
    const targetForm = document.getElementById(`${formType}Form`);
    if (targetForm) {
        targetForm.classList.add('active');
    }
    
    // Clear any existing messages
    const message = document.querySelector('.auth-message');
    if (message) message.remove();
}

function showForgotPassword() {
    switchAuthForm('forgot');
}

// Scroll to section
function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.scrollIntoView({ behavior: 'smooth' });
    }
}

// Initialize auth manager
const authManager = new AuthManager();

// Make functions globally available
window.openAuthModal = openAuthModal;
window.closeAuthModal = closeAuthModal;
window.switchAuthForm = switchAuthForm;
window.showForgotPassword = showForgotPassword;
window.scrollToSection = scrollToSection;
window.handleLogin = (e) => authManager.handleLogin(e);
window.handleSignup = (e) => authManager.handleSignup(e);
window.handleForgotPassword = (e) => authManager.handleForgotPassword(e);
window.logout = () => authManager.logout();