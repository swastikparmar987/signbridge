import path from "path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  cacheComponents: true,
  partialPrefetching: true,
  async rewrites() {
    const apiOrigin = (process.env.SIGNBRIDGE_API_URL || "http://127.0.0.1:8001").replace(/\/+$/, "");
    return [
      {
        source: "/",
        destination: "/landing.html",
      },
      {
        source: "/api/:path*",
        destination: `${apiOrigin}/api/:path*`,
      },
      {
        source: "/static/:path*",
        destination: `${apiOrigin}/static/:path*`,
      },
    ];
  },
  turbopack: {
    root: path.resolve(__dirname),
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;

