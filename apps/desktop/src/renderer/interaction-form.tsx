import { useState } from "react";
import { Button } from "./components/ui/radix/button.js";
import { Input } from "./components/ui/radix/input.js";
import { NativeSelect } from "./components/ui/radix/native-select.js";
import { t, useTranslation } from "./i18n.js";

type JsonObject = Record<string, unknown>;

export function NativeInteractionForm({
  id,
  title,
  schema,
  onRespond,
}: {
  id: string;
  title: string;
  schema: JsonObject;
  onRespond: (answer: unknown | undefined) => Promise<void>;
}) {
  useTranslation();
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string>();
  const respond = async (answer: unknown | undefined) => {
    setPending(true);
    setError(undefined);
    try {
      await onRespond(answer);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setPending(false);
    }
  };
  return (
    <form
      id={`native-interaction-${id}`}
      className="mx-auto mb-6 w-full max-w-3xl rounded-2xl border border-neutral-300 bg-neutral-50 p-5"
      onSubmit={(event) => {
        event.preventDefault();
        if (event.currentTarget.reportValidity())
          void respond(formPayload(new FormData(event.currentTarget), schema));
      }}
    >
      <div className="mb-1 text-xs text-neutral-500">{t("需要你的确认")}</div>
      <h3 className="mb-4 font-medium">{title}</h3>
      {typeof schema.description === "string" && (
        <pre className="mb-4 max-h-64 overflow-auto whitespace-pre-wrap break-words text-sm">
          {schema.description}
        </pre>
      )}
      <fieldset disabled={pending} className="grid gap-4">
        {Object.entries(object(schema.properties)).map(([name, value]) => (
          <InteractionField
            key={name}
            name={name}
            schema={object(value)}
            required={Array.isArray(schema.required) && schema.required.includes(name)}
          />
        ))}
        <div className="mt-1 flex gap-2">
          <Button type="submit">{pending ? t("正在提交…") : t("继续")}</Button>
          <Button variant="outline" type="button" onClick={() => void respond(undefined)}>
            {t("取消")}
          </Button>
        </div>
      </fieldset>
      {error !== undefined && (
        <p role="alert" className="mt-3 break-words text-sm">
          {error}
        </p>
      )}
    </form>
  );
}

function InteractionField({
  name,
  schema,
  required,
}: {
  name: string;
  schema: JsonObject;
  required: boolean;
}) {
  useTranslation();
  const options = choices(schema.type === "array" ? object(schema.items) : schema);
  const label = typeof schema.title === "string" ? schema.title : name;
  if (schema.type === "boolean") {
    return (
      <label className="flex items-center gap-2">
        <input name={name} type="checkbox" className="size-4 accent-neutral-900" />
        {label}
      </label>
    );
  }
  if (options.length > 0) {
    return (
      <label className="grid gap-1.5 text-sm">
        {label}
        <NativeSelect
          className={schema.type === "array" ? "h-auto min-h-24 [&+svg]:hidden" : undefined}
          multiple={schema.type === "array"}
          name={name}
          required={required}
          defaultValue={schema.type === "array" ? [] : ""}
        >
          {schema.type !== "array" && <option value="">{t("请选择")}</option>}
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </NativeSelect>
      </label>
    );
  }
  const type =
    schema.type === "number" || schema.type === "integer"
      ? "number"
      : ["date", "email", "url", "password"].includes(String(schema.format))
        ? String(schema.format)
        : "text";
  return (
    <label className="grid gap-1.5 text-sm">
      {label}
      <Input
        name={name}
        type={type}
        required={required}
        step={schema.type === "integer" ? 1 : "any"}
        min={typeof schema.minimum === "number" ? schema.minimum : undefined}
        max={typeof schema.maximum === "number" ? schema.maximum : undefined}
        minLength={typeof schema.minLength === "number" ? schema.minLength : undefined}
        maxLength={typeof schema.maxLength === "number" ? schema.maxLength : undefined}
      />
    </label>
  );
}

function formPayload(data: FormData, schema: JsonObject | undefined): JsonObject {
  const result: JsonObject = {};
  const required = new Set(Array.isArray(schema?.required) ? schema.required : []);
  for (const [name, field] of Object.entries(object(schema?.properties))) {
    const definition = object(field);
    if (definition.type === "boolean") {
      result[name] = data.has(name);
    } else if (definition.type === "array") {
      const values = data.getAll(name).map(String);
      if (values.length > 0 || required.has(name)) result[name] = values;
    } else {
      const value = data.get(name);
      if (value !== null && (String(value) !== "" || required.has(name))) {
        result[name] =
          definition.type === "number" || definition.type === "integer"
            ? Number(value)
            : String(value);
      }
    }
  }
  return result;
}

function choices(schema: JsonObject): Array<{ value: string; label: string }> {
  if (Array.isArray(schema.enum)) {
    return schema.enum.map((value) => ({ value: String(value), label: String(value) }));
  }
  if (!Array.isArray(schema.oneOf)) return [];
  return schema.oneOf.map((entry) => {
    const option = object(entry);
    return {
      value: String(option.const),
      label: typeof option.title === "string" ? option.title : String(option.const),
    };
  });
}

function object(value: unknown): JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as JsonObject)
    : {};
}
