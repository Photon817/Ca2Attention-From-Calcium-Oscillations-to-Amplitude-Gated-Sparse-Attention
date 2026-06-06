import { useState } from 'react';
import { ChevronDown, ChevronUp, MessageSquare, Eye } from 'lucide-react';
import type { PipelineKOL } from '../types';
import { KOLBadge } from './StatusBadge';
import Timeline from './Timeline';
import KOLDrawer from './KOLDrawer';

export default function PipelineTable({ rows }: { rows: PipelineKOL[] }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const [drawerKolId, setDrawerKolId] = useState<string | null>(null);

  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
      <table className="w-full text-sm">
        <thead className="bg-slate-50 text-slate-500">
          <tr>
            <th className="text-left px-4 py-3 font-medium">KOL</th>
            <th className="text-left px-4 py-3 font-medium">Status</th>
            <th className="text-left px-4 py-3 font-medium">报价</th>
            <th className="text-left px-4 py-3 font-medium">交付物</th>
            <th className="text-left px-4 py-3 font-medium">Deadline</th>
            <th className="text-left px-4 py-3 font-medium">最近动态</th>
            <th className="px-4 py-3"></th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {rows.map((r) => (
            <>
              <tr key={r.kol.id} className="hover:bg-slate-50">
                <td className="px-4 py-3">
                  <button onClick={() => setDrawerKolId(r.kol.id)} className="flex items-center gap-3 text-left group">
                    <img src={r.kol.avatar} alt="" className="w-8 h-8 rounded-full bg-slate-100" />
                    <div>
                      <div className="font-medium text-slate-900 group-hover:text-brand-600">{r.kol.name}</div>
                      <div className="text-xs text-slate-500">{r.kol.handle}</div>
                    </div>
                  </button>
                </td>
                <td className="px-4 py-3"><KOLBadge status={r.status} /></td>
                <td className="px-4 py-3 text-slate-700">¥{r.price.toLocaleString()}</td>
                <td className="px-4 py-3 text-slate-600">{r.deliverables.join('、')}</td>
                <td className="px-4 py-3 text-slate-600">{r.deadline}</td>
                <td className="px-4 py-3 text-slate-500 text-xs">{r.lastActivity}</td>
                <td className="px-4 py-3">
                  <button onClick={() => setExpanded(expanded === r.kol.id ? null : r.kol.id)} className="text-slate-400 hover:text-slate-600">
                    {expanded === r.kol.id ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                  </button>
                </td>
              </tr>
              {expanded === r.kol.id && (
                <tr>
                  <td colSpan={7} className="px-4 py-4 bg-slate-50">
                    <div className="mb-3"><Timeline current={r.status} /></div>
                    <div className="flex items-center gap-3">
                      <button onClick={() => setDrawerKolId(r.kol.id)} className="inline-flex items-center gap-1.5 text-xs font-medium text-brand-600 hover:text-brand-700">
                        <Eye size={14} /> 查看详情
                      </button>
                      <button className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-600 hover:text-slate-800">
                        <MessageSquare size={14} /> 添加备注
                      </button>
                    </div>
                  </td>
                </tr>
              )}
            </>
          ))}
        </tbody>
      </table>

      {drawerKolId && <KOLDrawer kolId={drawerKolId} onClose={() => setDrawerKolId(null)} />}
    </div>
  );
}