/** Locale namespace owned by the product annotation experience. */
export const ANNOTATION_LOCALE_NS = "swarmx.annotation";

export const zh = {
  "selection.add": "加入对话",
  "selection.noteLabel": "可选说明",
  "selection.notePlaceholder": "补充你希望模型关注的内容",
  "selection.confirm": "确认加入",
  "selection.cancel": "取消",
  "tray.countOne": "{count} 条批注",
  "tray.countMany": "{count} 条批注",
  "tray.dialog": "对话批注",
  "tray.selectedText": "选中文本：",
  "tray.edit": "编辑第 {index} 条批注",
  "tray.remove": "删除第 {index} 条批注",
  "tray.editLabel": "批注说明",
  "error.add": "无法加入批注，请稍后重试。",
  "error.limit": "每条消息最多可加入 {count} 条批注。",
  "presentation.fileCitation": "文件引用",
  "presentation.webCitation": "网页引用",
  "presentation.containerFileCitation": "容器文件引用",
  "presentation.filePath": "文件路径",
  "presentation.page": "第 {page} 页",
  "presentation.region": "区域 {region}",
  "presentation.imagePoint": "图片点位",
  "presentation.userMessage": "用户消息",
  "presentation.steeringMessage": "追问消息",
  "presentation.assistantMessage": "助手消息",
  "presentation.message": "消息 #{seq}",
  "presentation.annotation": "批注",
  "presentation.invalidOne": "已隐藏 1 条无效批注",
  "presentation.invalidMany": "已隐藏 {count} 条无效批注",
} as const;

export type AnnotationLocaleKey = keyof typeof zh;

export const en: Record<AnnotationLocaleKey, string> = {
  "selection.add": "Add to chat",
  "selection.noteLabel": "Optional note",
  "selection.notePlaceholder": "Add what you want the model to focus on",
  "selection.confirm": "Add annotation",
  "selection.cancel": "Cancel",
  "tray.countOne": "{count} annotation",
  "tray.countMany": "{count} annotations",
  "tray.dialog": "Conversation annotations",
  "tray.selectedText": "Selected text:",
  "tray.edit": "Edit annotation {index}",
  "tray.remove": "Remove annotation {index}",
  "tray.editLabel": "Annotation note",
  "error.add": "Couldn't add the annotation. Try again.",
  "error.limit": "A message can include up to {count} annotations.",
  "presentation.fileCitation": "File citation",
  "presentation.webCitation": "Web citation",
  "presentation.containerFileCitation": "Container file citation",
  "presentation.filePath": "File path",
  "presentation.page": "Page {page}",
  "presentation.region": "Region {region}",
  "presentation.imagePoint": "Image point",
  "presentation.userMessage": "User message",
  "presentation.steeringMessage": "Steering message",
  "presentation.assistantMessage": "Assistant message",
  "presentation.message": "Message #{seq}",
  "presentation.annotation": "Annotation",
  "presentation.invalidOne": "1 invalid annotation hidden",
  "presentation.invalidMany": "{count} invalid annotations hidden",
};

export type AnnotationTranslate = (
  key: AnnotationLocaleKey,
  values?: Readonly<Record<string, string | number>>,
) => string;

export function englishAnnotationText(
  key: AnnotationLocaleKey,
  values: Readonly<Record<string, string | number>> = {},
): string {
  return Object.entries(values).reduce(
    (text, [name, value]) => text.replaceAll(`{${name}}`, String(value)),
    en[key],
  );
}
