import { Link } from 'react-router-dom';
import { ChevronRight } from 'lucide-react';

export default function Nav({ items }: { items: { label: string; to?: string }[] }) {
  return (
    <nav className="flex items-center gap-2 text-sm text-slate-500 mb-6">
      {items.map((item, idx) => (
        <span key={idx} className="flex items-center gap-2">
          {idx > 0 && <ChevronRight size={14} />}
          {item.to ? (
            <Link to={item.to} className="hover:text-brand-600">{item.label}</Link>
          ) : (
            <span className="text-slate-900 font-medium">{item.label}</span>
          )}
        </span>
      ))}
    </nav>
  );
}