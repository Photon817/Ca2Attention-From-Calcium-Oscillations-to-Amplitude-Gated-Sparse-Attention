import { CheckCircle2, Circle, Clock } from 'lucide-react';

const steps = [
  { key: 'outreach', label: '触达' },
  { key: 'negotiating', label: '议价' },
  { key: 'contracted', label: '签约' },
  { key: 'content_creation', label: '内容制作' },
  { key: 'preview_review', label: 'Preview 审核' },
  { key: 'published', label: '发布' },
] as const;

export default function Timeline({ current }: { current: string }) {
  const idx = steps.findIndex((s) => s.key === current);
  return (
    <div className="flex items-center gap-1">
      {steps.map((s, i) => {
        const done = i <= idx;
        const isCurrent = i === idx;
        return (
          <div key={s.key} className="flex items-center gap-1">
            <div className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium ${
              isCurrent ? 'bg-brand-600 text-white' : done ? 'bg-brand-50 text-brand-700' : 'bg-slate-100 text-slate-400'
            }`}>
              {done ? (isCurrent ? <Clock size={12} /> : <CheckCircle2 size={12} />) : <Circle size={12} />}
              {s.label}
            </div>
            {i < steps.length - 1 && (
              <div className={`w-4 h-px ${done ? 'bg-brand-300' : 'bg-slate-200'}`} />
            )}
          </div>
        );
      })}
    </div>
  );
}