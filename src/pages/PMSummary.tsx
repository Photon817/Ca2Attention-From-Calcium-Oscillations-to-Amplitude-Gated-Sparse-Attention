import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Download, Sparkles } from 'lucide-react';
import Nav from '../components/Nav';
import { campaigns } from '../mockData';

export default function PMSummary() {
  const { id } = useParams<{ id: string }>();
  const campaign = campaigns.find((c) => c.id === id);
  if (!campaign) return <div className="text-slate-500">Campaign 不存在</div>;

  return (
    <div>
      <Nav items={[{ label: 'Campaigns', to: '/campaigns' }, { label: campaign.name, to: `/campaigns/${id}` }, { label: 'PM Summary' }]} />

      <div className="flex items-center justify-between mb-5">
        <h2 className="text-xl font-bold text-slate-900">PM Summary</h2>
        <div className="flex items-center gap-2">
          <Link to={`/campaigns/${id}`} className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg border border-slate-200 text-sm text-slate-600 hover:bg-slate-50">
            <ArrowLeft size={14} /> 返回
          </Link>
          <button className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg border border-slate-200 text-sm text-slate-600 hover:bg-slate-50">
            <Download size={14} /> 导出 PDF
          </button>
        </div>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm text-slate-500">Campaign</div>
            <div className="text-lg font-bold text-slate-900">{campaign.name}</div>
          </div>
          <span className="px-2 py-1 rounded bg-brand-50 text-brand-700 text-xs font-medium">AI Generated</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="rounded-lg bg-slate-50 p-3">
            <div className="text-xs text-slate-500">预算使用率</div>
            <div className="text-lg font-bold text-slate-900">{((campaign.spent / campaign.budget) * 100).toFixed(0)}%</div>
          </div>
          <div className="rounded-lg bg-slate-50 p-3">
            <div className="text-xs text-slate-500">目标 KOL</div>
            <div className="text-lg font-bold text-slate-900">{campaign.engagedKOLs}/{campaign.targetKOLs}</div>
          </div>
          <div className="rounded-lg bg-slate-50 p-3">
            <div className="text-xs text-slate-500">已发布内容</div>
            <div className="text-lg font-bold text-slate-900">8</div>
          </div>
          <div className="rounded-lg bg-slate-50 p-3">
            <div className="text-xs text-slate-500">预估曝光</div>
            <div className="text-lg font-bold text-slate-900">320W+</div>
          </div>
        </div>

        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-slate-900">阶段进展</h3>
          <div className="space-y-2 text-sm text-slate-700">
            <p>· 目前已有 12 位 KOL 进入合作流程，其中 8 位已完成发布，3 位在内容制作阶段，1 位在 Preview 审核。</p>
            <p>· 内容质量整体良好，平均互动率 4.2%，高于行业均值。</p>
            <p>· 存在 2 个预算风险项，建议与溢价 KOL 重新议价或启用备选名单。</p>
          </div>
        </div>

        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-slate-900">关键洞察</h3>
          <ul className="list-disc pl-4 text-sm text-slate-700 space-y-1">
            <li>Instagram 渠道 ROI 最高（1:4.3），建议下季度增加投入。</li>
            <li>头部 KOL 曝光稳定，但腰部 KOL 互动率反而更高，可考虑调整配比。</li>
            <li>用户对「成分解读」类内容反响最好，可作为后续内容策略方向。</li>
          </ul>
        </div>

        <div className="rounded-lg border border-brand-200 bg-brand-50 p-4">
          <div className="flex items-center gap-2 mb-2">
            <Sparkles size={16} className="text-brand-600" />
            <span className="text-sm font-semibold text-brand-800">AI 建议</span>
          </div>
          <p className="text-sm text-brand-900">建议提前 2 周启动下季度 Campaign 的 KOL 筛选，当前高互动率 KOL 档期紧张。</p>
        </div>
      </div>
    </div>
  );
}