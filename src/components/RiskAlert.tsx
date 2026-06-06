import { AlertTriangle, AlertCircle, Info } from 'lucide-react';
import type { RiskFlag } from '../types';
import { RiskBadge } from './StatusBadge';

const icons = {
  critical: AlertTriangle,
  warning: AlertCircle,
  info: Info,
};

export default function RiskAlert({ flags }: { flags: RiskFlag[] }) {
  if (!flags.length) return null;
  return (
    <div className="space-y-2">
      {flags.map((f) => {
        const Icon = icons[f.level];
        return (
          <div key={f.id} className="flex items-start gap-3 rounded-lg border border-red-100 bg-red-50 px-3 py-2">
            <Icon size={16} className="text-red-600 mt-0.5" />
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-red-800">{f.message}</span>
                <RiskBadge level={f.level} />
              </div>
              <div className="text-xs text-red-600 mt-0.5 capitalize">{f.category}</div>
            </div>
          </div>
        );
      })}
    </div>
  );
}