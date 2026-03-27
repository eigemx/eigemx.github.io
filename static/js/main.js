document.addEventListener('DOMContentLoaded', (event) => {
    const themeSwitcher = document.getElementById('theme-switcher');
    const htmlElement = document.documentElement;
    const bodyElement = document.body;
    const pathPrefix = bodyElement.dataset.pathPrefix || '';
    let pygmentsLink = document.querySelector('link[href*="pygments"]');

    // Ensure pygmentsLink exists and is in the head
    if (!pygmentsLink) {
        pygmentsLink = document.createElement('link');
        pygmentsLink.rel = 'stylesheet';
        document.head.appendChild(pygmentsLink);
    }

    // Function to set the theme
    const setTheme = (theme) => {
        
        if (theme === 'dark') {
            htmlElement.classList.add('dark-mode');
            themeSwitcher.innerHTML = '&#9788;'; // Sun icon
            pygmentsLink.href = `${pathPrefix}static/css/pygments-rose-pine.css`;
        } else {
            htmlElement.classList.remove('dark-mode');
            themeSwitcher.innerHTML = '&#9790;'; // Moon icon
            pygmentsLink.href = `${pathPrefix}static/css/pygments-rose-pine-dawn.css`;
        }
        localStorage.setItem('theme', theme);
    };

    // Function to get system preference
    const getSystemPreference = () => {
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            return 'dark';
        }
        return 'light';
    };

    // Call setTheme on initial load
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        setTheme(savedTheme);
    } else {
        // Use system preference if no saved theme
        setTheme(getSystemPreference());
    }

    // Listen for system theme changes
    if (window.matchMedia) {
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            // Only update if user hasn't explicitly set a theme
            if (!localStorage.getItem('theme')) {
                setTheme(e.matches ? 'dark' : 'light');
            }
        });
    }

    // Toggle theme on click
    themeSwitcher.addEventListener('click', () => {
        
        if (htmlElement.classList.contains('dark-mode')) {
            setTheme('light');
        } else {
            setTheme('dark');
        }
    });

    // Image zoom functionality
    const images = document.querySelectorAll('.content img');
    images.forEach(img => {
        const container = document.createElement('div');
        container.className = 'img-container';

        const zoomIcon = document.createElement('div');
        zoomIcon.className = 'zoom-icon';
        zoomIcon.innerHTML = '&#43;';

        img.parentNode.insertBefore(container, img);
        container.appendChild(img);
        container.appendChild(zoomIcon);

        container.addEventListener('click', () => {
            const overlay = document.createElement('div');
            overlay.className = 'img-overlay';
            
            const zoomedImg = document.createElement('img');
            zoomedImg.src = img.src;

            overlay.appendChild(zoomedImg);
            document.body.appendChild(overlay);

            overlay.style.display = 'flex';

            overlay.addEventListener('click', () => {
                document.body.removeChild(overlay);
            });
        });
    });
});