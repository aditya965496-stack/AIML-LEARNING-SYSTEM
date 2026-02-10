import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
  index("routes/home.tsx"),
  route("text-explanation", "routes/text-explanation.tsx"),
  route("code-generation", "routes/code-generation.tsx"),
  route("audio-learning", "routes/audio-learning.tsx"),
  route("image-visualization", "routes/image-visualization.tsx"),
] satisfies RouteConfig;
