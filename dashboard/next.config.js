/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  images: {
    domains: ['wandb.ai', 'api.wandb.ai'],
  },
  async rewrites() {
    return [
      {
        source: '/api/ml/:path*',
        destination: 'http://localhost:8000/:path*', // FastAPI backend
      },
    ];
  },
};

module.exports = nextConfig;
