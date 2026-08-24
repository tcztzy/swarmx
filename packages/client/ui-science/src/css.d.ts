declare module "*.module.css" {
  const classes: Readonly<Record<string, string>>;
  export default classes;
}

declare module "*?raw" {
  const source: string;
  export default source;
}
