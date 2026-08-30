export default function PromptChips() {
  const prompts = props.prompts || [];

  return (
    <div className="mt-3 flex w-full flex-col items-center gap-2">
      <p className="text-xs text-muted-foreground">Suggested next prompts</p>
      <div className="flex w-full flex-wrap justify-center gap-2">
        {prompts.map((item) => (
          <button
            key={item.query}
            type="button"
            onClick={() => sendUserMessage(item.query)}
            className="rounded-full border border-border bg-transparent px-3 py-1.5 text-sm text-foreground transition-colors hover:bg-muted"
          >
            {item.label}
          </button>
        ))}
      </div>
    </div>
  );
}
