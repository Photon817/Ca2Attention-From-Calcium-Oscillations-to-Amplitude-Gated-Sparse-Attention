import { BarChart3, Users, ClipboardCheck, AlertOctagon, CheckCircle2, ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import KpiCard from '../components/KpiCard';
import AiCard from '../components/AiCard';
import { RiskBadge } from '../components/StatusBadge';
import { kpiSnapshot, aiSummary, todos, campaigns } from '../mockData';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';

const statusData = [
  { name: '触达中', value: 3, color: '#94a3b8' },
  { name: '议价中', value: 2, color: '#3b82f6' },
  { name: '已签约', value: 4, color: '#6366f1' },
  { name: '内容制作', value: 5, color: '#a855f7' },
  { name: 'Preview 审核', value: 3, color: '#f59e0b' },
  { name: '已发布', value: 8, color: '#22c55e' },
];

export default function Dashboard() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-slate-900">Dashboard</h2>
        <p className="text-sm text-slate-500 mt-1">今日概览 · 2024-06-20</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard title="进行中 Campaign" value={kpiSnapshot.activeCampaigns} sub="本月目标 4 个" icon={BarChart3} tone="success" />
        <KpiCard title="在合作 KOL" value={kpiSnapshot.liveKOLs} sub="较上周 +3" icon={Users} tone="neutral" />
        <KpiCard title="待审核 Preview" value={kpiSnapshot.pendingReviews} sub="需 48h 内处理" icon={ClipboardCheck} tone="warning" />
        <KpiCard title="逾期任务" value={kpiSnapshot.overdueTasks} sub="请立即跟进" icon={AlertOctagon} tone="danger" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-xl border border-slate-200 p-5">
            <h3 className="text-sm font-semibold text-slate-900 mb-4">KOL 状态分布</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={statusData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                    {statusData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-white rounded-xl border border-slate-200 p-5">
            <h3 className="text-sm font-semibold text-slate-900 mb-3">风险提醒</h3>
            <div className="space-y-3">
              <div className="flex items-start justify-between rounded-lg border border-red-100 bg-red-50 px-3 py-2">
                <div>
                  <div className="text-sm font-medium text-red-800">TechBro_阿伟 报价超预算 25%</div>
                  <div className="text-xs text-red-600 mt-0.5">Campaign: 2024 Summer Skincare Launch</div>
                </div>
                <RiskBadge level="critical" />
              </div>
              <div className="flex items-start justify-between rounded-lg border border-amber-100 bg-amber-50 px-3 py-2">
                <div>
                  <div className="text-sm font-medium text-amber-800">LisaFitness Preview 缺少 CTA</div>
                  <div className="text-xs text-amber-600 mt-0.5">Campaign: 2024 Summer Skincare Launch</div>
                </div>
                <RiskBadge level="warning" />
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <AiCard title="AI 每日摘要">
            <div className="font-medium text-slate-900">{aiSummary.headline}</div>
            <ul className="list-disc pl-4 space-y-1">
              {aiSummary.points.map((p, i) => <li key={i}>{p}</li>)}
            </ul>
            <div className="pt-2 border-t border-brand-100 mt-2">
              <div className="text-brand-700 font-medium">建议行动</div>
              <div className="text-slate-600">{aiSummary.suggestedAction}</div>
            </div>
          </AiCard>

          <div className="bg-white rounded-xl border border-slate-200 p-5">
            <h3 className="text-sm font-semibold text-slate-900 mb-3">今日待办</h3>
            <div className="space-y-2">
              {todos.map((t) => (
                <div key={t.id} className="flex items-start gap-3">
                  <CheckCircle2 size={16} className={`mt-0.5 ${t.done ? 'text-green-500' : 'text-slate-300'}`} />
                  <div className="flex-1">
                    <div className={`text-sm ${t.done ? 'line-through text-slate-400' : 'text-slate-700'}`}>{t.title}</div>
                    <div className="text-xs text-slate-500">{t.campaignName} · {t.dueDate}</div>
                  </div>
                  {!t.done && (
                    <span className={`text-xs px-1.5 py-0.5 rounded ${
                      t.priority === 'high' ? 'bg-red-50 text-red-600' : t.priority === 'medium' ? 'bg-amber-50 text-amber-600' : 'bg-slate-50 text-slate-500'
                    }`}>{t.priority}</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 p-5">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-slate-900">快速跳转</h3>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {campaigns.map((c) => (
            <Link key={c.id} to={`/campaigns/${c.id}`} className="flex items-center justify-between rounded-lg border border-slate-100 bg-slate-50 px-4 py-3 hover:border-brand-200 hover:bg-brand-50 transition-colors">
              <div>
                <div className="text-sm font-medium text-slate-900">{c.name}</div>
                <div className="text-xs text-slate-500">{c.brand}</div>
              </div>
              <ArrowRight size={14} className="text-slate-400" />
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}