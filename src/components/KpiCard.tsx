import type { LucideIcon } from 'lucide-react';

export default function KpiCard({ title, value, sub, icon: Icon, tone = 'neutral' }: {
  title: string;
  value: string | number;
  sub?: string;
  icon: LucideIcon;
  tone?: 'neutral' | 'success' | 'warning' | 'danger';
}) {
  const toneStyles = {
    neutral: 'bg-white border-slate-200',
    success: 'bg-white border-green-200',
    warning: 'bg-white border-amber-200',
    danger: 'bg-white border-red-200',
  };
  const iconTones = {
    neutral: 'text-slate-500 bg-slate-50',
    success: 'text-green-600 bg-green-50',
    warning: 'text-amber-600 bg-amber-50',
    danger: 'text-red-600 bg-red-50',
  };

  return (
    <div className={`rounded-xl border p-5 ${toneStyles[tone]}`}>
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-slate-500">{title}</span>
        <div className={`p-2 rounded-lg ${iconTones[tone]}`}>
          <Icon size={18} />
        </div>
      </div>
      <div className="text-2xl font-bold text-slate-900">{value}</div>
      {sub && <div className="text-xs text-slate-500 mt-1">{sub}</div>}
    </div>
  );
}