// script.js - Main Application Logic for Landing Page
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    setupNavigation();
    setupScrollEffects();
    setupAnimations();
    setupParallax();
    setupServiceHealthCheck();
}

function setupNavigation() {
    const navbar = document.querySelector('.navbar');
    const hamburger = document.querySelector('.hamburger');
    const navLinks = document.querySelectorAll('.nav-link');
    const navMenu = document.querySelector('.nav-menu');

    // Navbar scroll effect
    window.addEventListener('scroll', () => {
        if (window.scrollY > 100) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });

    // Hamburger menu (for mobile)
    if (hamburger && navMenu) {
        hamburger.addEventListener('click', () => {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
        });
    }

    // Smooth scroll for navigation links
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);
            const targetSection = document.getElementById(targetId);
            
            if (targetSection) {
                targetSection.scrollIntoView({ behavior: 'smooth' });
                
                // Update active link
                navLinks.forEach(l => l.classList.remove('active'));
                link.classList.add('active');

                // Close mobile menu if open
                if (navMenu && navMenu.classList.contains('active')) {
                    navMenu.classList.remove('active');
                    hamburger.classList.remove('active');
                }
            }
        });
    });

    // Close mobile menu when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.nav-menu') && !e.target.closest('.hamburger')) {
            if (navMenu) navMenu.classList.remove('active');
            if (hamburger) hamburger.classList.remove('active');
        }
    });
}

function setupScrollEffects() {
    // Reveal animations on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
            }
        });
    }, observerOptions);

    // Observe elements for reveal animation
    document.querySelectorAll('.feature-card, .tech-item').forEach(el => {
        el.classList.add('reveal-item');
        observer.observe(el);
    });
}

function setupAnimations() {
    // Add CSS for reveal animations
    const style = document.createElement('style');
    style.textContent = `
        .reveal-item {
            opacity: 0;
            transform: translateY(30px);
            transition: all 0.6s ease;
        }
        
        .reveal-item.revealed {
            opacity: 1;
            transform: translateY(0);
        }
        
        .nav-menu.active {
            display: flex !important;
            flex-direction: column;
            position: absolute;
            top: 100%;
            left: 0;
            width: 100%;
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            padding: 2rem;
            border-top: 1px solid var(--glass-border);
        }
        
        .theme-transition * {
            transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease !important;
        }

        .hamburger.active span:nth-child(1) {
            transform: rotate(45deg) translate(6px, 6px);
        }

        .hamburger.active span:nth-child(2) {
            opacity: 0;
        }

        .hamburger.active span:nth-child(3) {
            transform: rotate(-45deg) translate(6px, -6px);
        }

        .user-dropdown.active {
            opacity: 1;
            visibility: visible;
            transform: translateY(0);
        }

        /* Enhanced severity indicators */
        .severity-level {
            padding: 0.25rem 1rem;
            border-radius: 15px;
            text-transform: uppercase;
            font-size: 0.875rem;
            font-weight: 600;
        }

        .severity-level.low {
            background: rgba(34, 197, 94, 0.15);
            color: #16a34a;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }

        .severity-level.medium {
            background: rgba(245, 158, 11, 0.15);
            color: #d97706;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }

        .severity-level.high {
            background: rgba(239, 68, 68, 0.15);
            color: #dc2626;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        /* Guidelines styling */
        .guidelines {
            border-left: 4px solid var(--primary-color);
        }

        .guidelines h4 {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }

        .guidelines ul {
            margin: 0;
            padding-left: 1.2rem;
        }

        .guidelines li {
            padding: 0.25rem 0;
            color: var(--text-secondary);
        }

        /* Loading states */
        .btn-loading {
            opacity: 0.7;
            pointer-events: none;
        }

        /* Error message styling */
        .error-message {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.2);
            color: #dc2626;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }

        .warning-message {
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid rgba(245, 158, 11, 0.2);
            color: #d97706;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
    `;
    document.head.appendChild(style);
}

function setupParallax() {
    // Simple parallax effect for hero background
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        const hero = document.querySelector('.hero-background');
        if (hero) {
            hero.style.transform = `translateY(${scrolled * 0.5}px)`;
        }
    });
}

function setupServiceHealthCheck() {
    // Health check for services
    async function checkServicesHealth() {
        try {
            // Check auth service
            const authResponse = await fetch('http://localhost:3000/api/auth/health');
            const authHealth = await authResponse.json();
            console.log('🔐 Auth service:', authHealth.status);

            // Check AI service
            const aiResponse = await fetch('http://localhost:5000/api/health');
            const aiHealth = await aiResponse.json();
            console.log('🤖 AI service:', aiHealth.status);
            
            // Update UI based on service status
            updateServiceStatusIndicators(authHealth, aiHealth);
        } catch (error) {
            console.warn('⚠️ Some services might be unavailable:', error);
            updateServiceStatusIndicators(null, null, error);
        }
    }

    function updateServiceStatusIndicators(authHealth, aiHealth, error = null) {
        // You can add visual indicators in the UI if needed
        if (error) {
            console.log('🌐 Services: Some services offline');
        } else {
            console.log('🌐 Services: All services running');
        }
    }

    // Initialize service health check
    setTimeout(checkServicesHealth, 2000);
    
    // Check every 30 seconds
    setInterval(checkServicesHealth, 30000);
}

// Enhanced notification system
window.showNotification = function(message, type = 'info', duration = 5000) {
    // Remove existing notifications
    const existingNotification = document.querySelector('.global-notification');
    if (existingNotification) {
        existingNotification.remove();
    }

    const notification = document.createElement('div');
    notification.className = `global-notification ${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${getNotificationIcon(type)}"></i>
            <span>${message}</span>
            <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;

    // Add styles for notification
    if (!document.querySelector('#notification-styles')) {
        const style = document.createElement('style');
        style.id = 'notification-styles';
        style.textContent = `
            .global-notification {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 10000;
                min-width: 300px;
                max-width: 500px;
                border-radius: 10px;
                padding: 1rem;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
                animation: slideInRight 0.3s ease;
            }
            
            .global-notification.info {
                background: var(--primary-color);
                color: white;
            }
            
            .global-notification.success {
                background: #10b981;
                color: white;
            }
            
            .global-notification.warning {
                background: #f59e0b;
                color: white;
            }
            
            .global-notification.error {
                background: #ef4444;
                color: white;
            }
            
            .notification-content {
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }
            
            .notification-close {
                background: none;
                border: none;
                color: inherit;
                cursor: pointer;
                margin-left: auto;
            }
            
            @keyframes slideInRight {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
        `;
        document.head.appendChild(style);
    }

    document.body.appendChild(notification);

    // Auto remove after duration
    if (duration > 0) {
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, duration);
    }

    return notification;
};

function getNotificationIcon(type) {
    const icons = {
        'info': 'info-circle',
        'success': 'check-circle',
        'warning': 'exclamation-triangle',
        'error': 'exclamation-circle'
    };
    return icons[type] || 'info-circle';
}

// Enhanced API call utility
window.makeApiCall = async (endpoint, options = {}) => {
    const token = localStorage.getItem('agriverse-token');
    
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            ...(token && { 'Authorization': `Bearer ${token}` })
        }
    };

    try {
        const response = await fetch(endpoint, { ...defaultOptions, ...options });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        
        // Show user-friendly error message
        if (error.message.includes('Failed to fetch')) {
            showNotification('Network error: Unable to connect to server', 'error', 5000);
        } else {
            showNotification(`Request failed: ${error.message}`, 'error', 5000);
        }
        
        throw error;
    }
};

// Utility function for form validation
window.validateForm = function(formData, rules) {
    const errors = {};
    
    for (const field in rules) {
        const value = formData[field];
        const fieldRules = rules[field];
        
        if (fieldRules.required && (!value || value.toString().trim() === '')) {
            errors[field] = `${field} is required`;
            continue;
        }
        
        if (fieldRules.min && value < fieldRules.min) {
            errors[field] = `${field} must be at least ${fieldRules.min}`;
        }
        
        if (fieldRules.max && value > fieldRules.max) {
            errors[field] = `${field} must be at most ${fieldRules.max}`;
        }
        
        if (fieldRules.pattern && !fieldRules.pattern.test(value)) {
            errors[field] = fieldRules.message || `${field} format is invalid`;
        }
    }
    
    return {
        isValid: Object.keys(errors).length === 0,
        errors
    };
};

// Export for use in other files
window.utils = {
    showNotification,
    makeApiCall,
    validateForm
};