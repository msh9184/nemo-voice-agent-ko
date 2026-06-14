// import { defineConfig } from 'vite';
// import react from '@vitejs/plugin-react-swc';

// export default defineConfig({
//     plugins: [react()],
//     server: {
//         host: '0.0.0.0',  // Bind to all interfaces
//         port: 5173,       // Back to default Vite port
//         proxy: {
//             // Proxy /api requests to the backend server
//             '/connect': {
//                 target: 'http://0.0.0.0:7860', // Replace with your backend URL if needed
//                 changeOrigin: true,
//             },
//         },
//     },
// });

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';
import fs from 'fs';

// Check if SSL certificates exist for HTTPS
const keyPath = './key.pem';
const certPath = './cert.pem';
const hasSSL = fs.existsSync(keyPath) && fs.existsSync(certPath);

export default defineConfig({
    plugins: [react()],
    server: {
        host: '0.0.0.0',
        port: 5173,
        // Use polling instead of inotify to avoid ENOSPC error
        // (Linux file watcher limit exceeded)
        watch: {
            usePolling: true,
            interval: 1000,
        },
        ...(hasSSL && {
            https: {
                key: fs.readFileSync(keyPath),
                cert: fs.readFileSync(certPath),
            },
        }),
        proxy: {
            '/connect': {
                target: 'http://0.0.0.0:7860',
                changeOrigin: true,
            },
        },
    },
});
