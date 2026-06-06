import { X, Mail, MapPin, Users, BarChart3, StickyNote } from 'lucide-react';
import { kols } from '../mockData';

export default function KOLDrawer({ kolId, onClose }: { kolId: string; onClose: () => void }) {
  const kol = kols.find((k) => k.id === kolId);
  if (!kol) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/30" onClick={onClose} />
      <div className="relative w-full max-w-md bg-white h-full shadow-2xl flex flex-col">
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-100">
          <h2 className="text-base font-semibold">KOL 详情</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600"><X size={20} /></button>
        </div>
        <div className="flex-1 overflow-auto p-5 space-y-6">
          <div className="flex items-center gap-4">
            <img src={kol.avatar} alt="" className="w-16 h-16 rounded-full bg-slate-100" />
            <div>
              <div className="text-lg font-bold text-slate-900">{kol.name}</div>
              <div className="text-sm text-slate-500">{kol.handle} · {kol.platform}</div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-lg bg-slate-50 p-3">
              <div className="flex items-center gap-1.5 text-xs text-slate-500 mb-1"><Users size={14} /> 粉丝</div>
              <div className="text-sm font-semibold text-slate-900">{kol.followers.toLocaleString()}</div>
            </div>
            <div className="rounded-lg bg-slate-50 p-3">
              <div className="flex items-center gap-1.5 text-xs text-slate-500 mb-1"><BarChart3 size={14} /> 互动率</div>
              <div className="text-sm font-semibold text-slate-900">{(kol.engagementRate * 100).toFixed(1)}%</div>
            </div>
          </div>

          <div className="space-y-2 text-sm text-slate-700">
            <div className="flex items-center gap-2"><MapPin size={14} className="text-slate-400" /> {kol.location}</div>
            <div className="flex items-center gap-2"><Mail size={14} className="text-slate-400" /> {kol.email}</div>
            <div>领域：{kol.category}</div>
            <div>报价区间：{kol.priceRange}</div>
          </div>

          <div>
            <h3 className="text-sm font-semibold text-slate-900 mb-2 flex items-center gap-1.5"><StickyNote size={14} /> 备注</h3>
            <div className="space-y-2">
              {kol.notes.length === 0 && <div className="text-xs text-slate-400">暂无备注</div>}
              {kol.notes.map((n) => (
                <div key={n.id} className="rounded-lg border border-slate-100 bg-slate-50 p-3">
                  <div className="text-xs text-slate-500 mb-1">{n.author} · {new Date(n.createdAt).toLocaleDateString()}</div>
                  <div className="text-sm text-slate-700">{n.content}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}