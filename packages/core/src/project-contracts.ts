import { z } from "zod";

export const ProjectDataSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  cwd: z.string().min(1),
  pinned: z.boolean().default(false),
  createdAt: z.string(),
  updatedAt: z.string(),
  removedAt: z.string().optional(),
});

export type ProjectData = z.infer<typeof ProjectDataSchema>;
