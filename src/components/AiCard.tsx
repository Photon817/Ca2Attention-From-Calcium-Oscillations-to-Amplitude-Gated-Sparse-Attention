import { Sparkles } from 'lucide-react';

export default function AiCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-brand-200 bg-brand-50/50 p-5">
      <div className="flex items-center gap-2 mb-3">
        <Sparkles size={16} className="text-brand-600" />
        <h3 className="text-sm font-semibold text-brand-800">{title}</h3>
      </div>
      <div className="text-sm text-slate-700 space-y-2">{children}</div>
    </div>
  );
}