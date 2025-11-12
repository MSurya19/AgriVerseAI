const express = require('express');
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const nodemailer = require('nodemailer');
const cors = require('cors');
require('dotenv').config();

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// MongoDB Connection
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/AgriverseAI';
mongoose.connect(MONGODB_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
})
.then(() => console.log('✅ MongoDB connected successfully'))
.catch(err => console.error('❌ MongoDB connection error:', err));

// User Schema
const userSchema = new mongoose.Schema({
    firstName: {
        type: String,
        required: true,
        trim: true
    },
    lastName: {
        type: String,
        required: true,
        trim: true
    },
    email: {
        type: String,
        required: true,
        unique: true,
        lowercase: true,
        trim: true
    },
    password: {
        type: String,
        required: true,
        minlength: 6
    },
    role: {
        type: String,
        enum: ['farmer', 'researcher', 'admin'],
        default: 'farmer'
    },
    farmSize: {
        type: Number,
        min: 0
    },
    location: {
        type: String,
        trim: true
    },
    crops: [{
        type: String,
        trim: true
    }],
    preferences: {
        notifications: {
            type: Boolean,
            default: true
        },
        theme: {
            type: String,
            enum: ['light', 'dark'],
            default: 'light'
        }
    },
    lastLogin: {
        type: Date,
        default: Date.now
    },
    predictionCount: {
        type: Number,
        default: 0
    }
}, {
    timestamps: true
});

// Hash password before saving
userSchema.pre('save', async function(next) {
    if (!this.isModified('password')) return next();
    
    try {
        const salt = await bcrypt.genSalt(12);
        this.password = await bcrypt.hash(this.password, salt);
        next();
    } catch (error) {
        next(error);
    }
});

// Compare password method
userSchema.methods.comparePassword = async function(candidatePassword) {
    return await bcrypt.compare(candidatePassword, this.password);
};

// Update last login
userSchema.methods.updateLastLogin = async function() {
    this.lastLogin = new Date();
    return await this.save();
};

// Get user profile (without sensitive data)
userSchema.methods.getProfile = function() {
    return {
        id: this._id,
        firstName: this.firstName,
        lastName: this.lastName,
        email: this.email,
        role: this.role,
        farmSize: this.farmSize,
        location: this.location,
        crops: this.crops,
        preferences: this.preferences,
        lastLogin: this.lastLogin,
        predictionCount: this.predictionCount,
        createdAt: this.createdAt
    };
};

// Static method to find by email
userSchema.statics.findByEmail = function(email) {
    return this.findOne({ email: email.toLowerCase() });
};

const User = mongoose.model('User', userSchema);

// JWT Secret
const JWT_SECRET = process.env.JWT_SECRET || 'agriverse-ai-super-secret-key-2024';

// Email transporter (configure for production)
const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
        user: process.env.EMAIL_USER,
        pass: process.env.EMAIL_PASS
    }
});

// Auth Middleware
const authMiddleware = async (req, res, next) => {
    try {
        const token = req.header('Authorization')?.replace('Bearer ', '');
        
        if (!token) {
            return res.status(401).json({ 
                success: false,
                message: 'No authentication token provided' 
            });
        }

        const decoded = jwt.verify(token, JWT_SECRET);
        const user = await User.findById(decoded.userId).select('-password');
        
        if (!user) {
            return res.status(401).json({ 
                success: false,
                message: 'Invalid authentication token' 
            });
        }

        req.user = user;
        next();
    } catch (error) {
        res.status(401).json({ 
            success: false,
            message: 'Token is not valid' 
        });
    }
};

// Routes

// Health check
app.get('/api/auth/health', (req, res) => {
    res.json({ 
        success: true,
        status: 'Auth server running', 
        timestamp: new Date().toISOString(),
        database: mongoose.connection.readyState === 1 ? 'connected' : 'disconnected',
        version: '2.0.0'
    });
});

// User registration
app.post('/api/auth/signup', async (req, res) => {
    try {
        const { firstName, lastName, email, password, role, farmSize, location, crops } = req.body;

        // Validate required fields
        if (!firstName || !lastName || !email || !password) {
            return res.status(400).json({
                success: false,
                message: 'All required fields must be provided'
            });
        }

        // Check if user already exists
        const existingUser = await User.findByEmail(email);
        if (existingUser) {
            return res.status(400).json({
                success: false,
                message: 'User already exists with this email address'
            });
        }

        // Create new user
        const user = new User({
            firstName,
            lastName,
            email,
            password,
            role: role || 'farmer',
            farmSize,
            location,
            crops
        });

        await user.save();

        // Generate JWT token
        const token = jwt.sign(
            { userId: user._id }, 
            JWT_SECRET, 
            { expiresIn: '7d' }
        );

        // Update last login
        await user.updateLastLogin();

        res.status(201).json({
            success: true,
            message: 'User registered successfully',
            token,
            user: user.getProfile()
        });

    } catch (error) {
        console.error('Signup error:', error);
        
        if (error.name === 'ValidationError') {
            const errors = Object.values(error.errors).map(err => err.message);
            return res.status(400).json({
                success: false,
                message: 'Validation failed',
                errors
            });
        }
        
        res.status(500).json({
            success: false,
            message: 'Server error during registration'
        });
    }
});

// User login
app.post('/api/auth/login', async (req, res) => {
    try {
        const { email, password } = req.body;

        // Validate input
        if (!email || !password) {
            return res.status(400).json({
                success: false,
                message: 'Email and password are required'
            });
        }

        // Find user
        const user = await User.findByEmail(email);
        if (!user) {
            return res.status(400).json({
                success: false,
                message: 'Invalid email or password'
            });
        }

        // Check password
        const isPasswordValid = await user.comparePassword(password);
        if (!isPasswordValid) {
            return res.status(400).json({
                success: false,
                message: 'Invalid email or password'
            });
        }

        // Generate JWT token
        const token = jwt.sign(
            { userId: user._id },
            JWT_SECRET,
            { expiresIn: '7d' }
        );

        // Update last login
        await user.updateLastLogin();

        res.json({
            success: true,
            message: 'Login successful',
            token,
            user: user.getProfile()
        });

    } catch (error) {
        console.error('Login error:', error);
        res.status(500).json({
            success: false,
            message: 'Server error during login'
        });
    }
});

// Get current user profile
app.get('/api/auth/me', authMiddleware, async (req, res) => {
    res.json({
        success: true,
        user: req.user.getProfile()
    });
});

// Update user profile
app.put('/api/auth/profile', authMiddleware, async (req, res) => {
    try {
        const { firstName, lastName, farmSize, location, crops, preferences } = req.body;
        
        const allowedUpdates = ['firstName', 'lastName', 'farmSize', 'location', 'crops', 'preferences'];
        const updates = {};
        
        allowedUpdates.forEach(field => {
            if (req.body[field] !== undefined) {
                updates[field] = req.body[field];
            }
        });

        const user = await User.findByIdAndUpdate(
            req.user._id,
            updates,
            { new: true, runValidators: true }
        ).select('-password');

        res.json({
            success: true,
            message: 'Profile updated successfully',
            user: user.getProfile()
        });

    } catch (error) {
        console.error('Profile update error:', error);
        res.status(500).json({
            success: false,
            message: 'Server error during profile update'
        });
    }
});

// Change password
app.put('/api/auth/change-password', authMiddleware, async (req, res) => {
    try {
        const { currentPassword, newPassword } = req.body;

        if (!currentPassword || !newPassword) {
            return res.status(400).json({
                success: false,
                message: 'Current password and new password are required'
            });
        }

        if (newPassword.length < 6) {
            return res.status(400).json({
                success: false,
                message: 'New password must be at least 6 characters long'
            });
        }

        const user = await User.findById(req.user._id);
        const isCurrentPasswordValid = await user.comparePassword(currentPassword);

        if (!isCurrentPasswordValid) {
            return res.status(400).json({
                success: false,
                message: 'Current password is incorrect'
            });
        }

        user.password = newPassword;
        await user.save();

        res.json({
            success: true,
            message: 'Password changed successfully'
        });

    } catch (error) {
        console.error('Password change error:', error);
        res.status(500).json({
            success: false,
            message: 'Server error during password change'
        });
    }
});

// Forgot password
app.post('/api/auth/forgot-password', async (req, res) => {
    try {
        const { email } = req.body;

        if (!email) {
            return res.status(400).json({
                success: false,
                message: 'Email is required'
            });
        }

        const user = await User.findByEmail(email);
        
        // Always return success to prevent email enumeration
        if (!user) {
            return res.json({
                success: true,
                message: 'If an account with that email exists, a reset link has been sent'
            });
        }

        // Generate reset token
        const resetToken = jwt.sign(
            { userId: user._id, purpose: 'password_reset' },
            JWT_SECRET,
            { expiresIn: '1h' }
        );

        // In a real application, you would:
        // 1. Save the reset token to the user document
        // 2. Send an email with a reset link
        // 3. Implement rate limiting
        
        console.log(`Password reset token for ${email}: ${resetToken}`);

        res.json({
            success: true,
            message: 'If an account with that email exists, a reset link has been sent'
        });

    } catch (error) {
        console.error('Forgot password error:', error);
        res.status(500).json({
            success: false,
            message: 'Server error during password reset'
        });
    }
});

// Get user statistics
app.get('/api/auth/stats', authMiddleware, async (req, res) => {
    try {
        const totalUsers = await User.countDocuments();
        const farmersCount = await User.countDocuments({ role: 'farmer' });
        const researchersCount = await User.countDocuments({ role: 'researcher' });
        
        res.json({
            success: true,
            stats: {
                totalUsers,
                farmersCount,
                researchersCount,
                adminCount: await User.countDocuments({ role: 'admin' })
            }
        });

    } catch (error) {
        console.error('Stats error:', error);
        res.status(500).json({
            success: false,
            message: 'Server error fetching statistics'
        });
    }
});

// Update prediction count
app.post('/api/auth/update-prediction-count', authMiddleware, async (req, res) => {
    try {
        const user = await User.findById(req.user._id);
        user.predictionCount = (user.predictionCount || 0) + 1;
        await user.save();

        res.json({
            success: true,
            predictionCount: user.predictionCount
        });
    } catch (error) {
        console.error('Update prediction count error:', error);
        res.status(500).json({
            success: false,
            message: 'Server error updating prediction count'
        });
    }
});

// Error handling middleware
app.use((error, req, res, next) => {
    console.error('Unhandled error:', error);
    res.status(500).json({
        success: false,
        message: 'Internal server error'
    });
});

// 404 handler
app.use('*', (req, res) => {
    res.status(404).json({
        success: false,
        message: 'API endpoint not found'
    });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`🔐 AgriVerseAI Auth Server v2.0 running on port ${PORT}`);
    console.log(`📊 MongoDB: ${MONGODB_URI}`);
    console.log(`🔑 JWT Secret: ${JWT_SECRET ? 'Set' : 'Not set'}`);
    console.log(`🌐 Health check: http://localhost:${PORT}/api/auth/health`);
});