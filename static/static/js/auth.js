$(document).ready(function() {
    // Login form handler
    $('#loginForm').submit(function(e) {
        e.preventDefault();
        
        const username = $('#username').val().trim();
        const password = $('#password').val();
        
        if (!username || !password) {
            showAlert('Please fill in all fields', 'danger');
            return;
        }
        
        showLoading();
        
        $.ajax({
            url: '/api/login',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                username: username,
                password: password
            }),
            success: function(response) {
                hideLoading();
                if (response.success) {
                    showAlert('Login successful! Redirecting...', 'success');
                    setTimeout(function() {
                        window.location.href = '/';
                    }, 1500);
                } else {
                    showAlert(response.message, 'danger');
                }
            },
            error: function(xhr, status, error) {
                hideLoading();
                showAlert('Login failed. Please try again.', 'danger');
            }
        });
    });
    
    // Register form handler
    $('#registerForm').submit(function(e) {
        e.preventDefault();
        
        const username = $('#username').val().trim();
        const email = $('#email').val().trim();
        const password = $('#password').val();
        const confirmPassword = $('#confirmPassword').val();
        
        // Validation
        if (!username || !email || !password || !confirmPassword) {
            showAlert('Please fill in all fields', 'danger');
            return;
        }
        
        if (password !== confirmPassword) {
            showAlert('Passwords do not match', 'danger');
            return;
        }
        
        if (password.length < 6) {
            showAlert('Password must be at least 6 characters long', 'danger');
            return;
        }
        
        if (!isValidEmail(email)) {
            showAlert('Please enter a valid email address', 'danger');
            return;
        }
        
        showLoading();
        
        $.ajax({
            url: '/api/register',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                username: username,
                email: email,
                password: password
            }),
            success: function(response) {
                hideLoading();
                if (response.success) {
                    showAlert('Registration successful! Redirecting to login...', 'success');
                    setTimeout(function() {
                        window.location.href = '/login';
                    }, 2000);
                } else {
                    showAlert(response.message, 'danger');
                }
            },
            error: function(xhr, status, error) {
                hideLoading();
                showAlert('Registration failed. Please try again.', 'danger');
            }
        });
    });
    
    // Helper functions
    function showAlert(message, type) {
        const alertHtml = `
            <div class="alert alert-${type} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="close" data-dismiss="alert" aria-label="Close">
                    <span aria-hidden="true">&times;</span>
                </button>
            </div>
        `;
        $('#alertContainer').html(alertHtml);
    }
    
    function showLoading() {
        $('body').append(`
            <div class="loading-overlay">
                <div class="loading-content">
                    <div class="spinner-border text-primary" role="status">
                        <span class="sr-only">Loading...</span>
                    </div>
                    <p class="mt-3">Please wait...</p>
                </div>
            </div>
        `);
        $('.loading-overlay').css('display', 'flex');
    }
    
    function hideLoading() {
        $('.loading-overlay').remove();
    }
    
    function isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }
});