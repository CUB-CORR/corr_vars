// Enhanced copy button functionality for CORR-Vars documentation

document.addEventListener('DOMContentLoaded', function() {
    // Wait for the copy button to be initialized by sphinx-copybutton
    setTimeout(function() {
        // Find all copy buttons
        const copyButtons = document.querySelectorAll('.copybtn');
        
        copyButtons.forEach(function(button) {
            // Enhance the copy button text
            if (button.textContent.trim() === 'Copy') {
                button.textContent = 'Copy';
            }
            
            // Add enhanced click feedback
            button.addEventListener('click', function() {
                const originalContent = button.textContent;
                button.textContent = 'Copied!';
                button.classList.add('success');
                
                // Reset after 2 seconds
                setTimeout(function() {
                    button.textContent = originalContent;
                    button.classList.remove('success');
                }, 2000);
            });
            
            // Add tooltip on hover
            button.title = 'Copy code to clipboard';
            
            // Ensure button has proper styling
            button.setAttribute('aria-label', 'Copy code to clipboard');
        });
        
        // Add keyboard shortcut (Ctrl+C when hovering over code block)
        document.querySelectorAll('div.highlight').forEach(function(codeBlock) {
            codeBlock.addEventListener('keydown', function(e) {
                if ((e.ctrlKey || e.metaKey) && e.key === 'c') {
                    const copyBtn = codeBlock.querySelector('.copybtn');
                    if (copyBtn && document.activeElement === codeBlock) {
                        e.preventDefault();
                        copyBtn.click();
                    }
                }
            });
            
            // Make code blocks focusable for keyboard navigation
            codeBlock.setAttribute('tabindex', '0');
        });
        
    }, 500); // Wait for sphinx-copybutton to initialize
});

// Custom function to handle special cases in CORR-Vars code
function cleanCorrVarsCode(text) {
    // Remove Python prompts and output indicators
    return text
        .replace(/^>>> /gm, '')          // Remove Python prompts
        .replace(/^... /gm, '')          // Remove continuation prompts
        .replace(/^\$ /gm, '')           // Remove shell prompts
        .replace(/^# /gm, '# ')          // Keep comments as-is
        .replace(/^\d+\s+/gm, '');       // Remove line numbers if present
}