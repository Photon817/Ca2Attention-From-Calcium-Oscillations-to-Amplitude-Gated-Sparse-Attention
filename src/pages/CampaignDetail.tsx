import { useParams, Link } from 'react-router-dom';
import { FileText, Send, Sparkles } from 'lucide-react';
import Nav from '../components/Nav';
import PipelineTable from '../components/PipelineTable';
import AiCard from '../components/AiCard';
import RiskAlert from '../components/RiskAlert';
import { campaigns, pipelines } from '../mockData';

export default function CampaignDetail() {
  const { id } = useParams<{ id: string }>();
  const campaign = campaigns.find((c) => c.id === id);
  const pipeline = pipelines[id || ''] || [];

  if (!campaign) return <div className="text-slate-500">Campaign 不存在</div>;

  return (
    <div>
      <Nav items={[{ label: 'Campaigns', to: '/campaigns' }, { label: campaign.name }]} />

      <div className="flex items-start justify-between mb-5">
        <div>
          <h2 className="text-xl font-bold text-slate-900">{campaign.name}</h2>
          <div className="text-sm text-slate-500 mt-1">{campaign.brand} · 负责人 {campaign.manager}</div>
        </div>
        <div className="flex items-center gap-2">
          <Link to={`/campaigns/${campaign.id}/summary`} className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg border border-slate-200 text-sm text-slate-600 hover:bg-slate-50">
            <FileText size={14} /> PM Summary
          </Link>
          <button className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg bg-brand-600 text-white text-sm font-medium hover:bg-brand-700">
            <Send size={14} /> 批量催稿
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-xl border border-slate-200 p-5 space-y-3">
            <h3 className="text-sm font-semibold text-slate-900">基本信息</h3>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="text-slate-500">预算: <span className="text-slate-900">¥{campaign.budget.toLocaleString()}</span></div>
              <div className="text-slate-500">已用: <span className="text-slate-900">¥{campaign.spent.toLocaleString()}</span></div>
              <div className="text-slate-500">目标 KOL: <span className="text-slate-900">{campaign.targetKOLs}</span></div>
              <div className="text-slate-500">已合作: <span className="text-slate-900">{campaign.engagedKOLs}</span></div>
            </div>
            <p className="text-sm text-slate-600 leading-relaxed">{campaign.description}</p>
            <div className="flex flex-wrap gap-2">
              {campaign.objectives.map((o) => (
                <span key={o} className="inline-flex items-center px-2 py-1 rounded-md bg-slate-50 text-slate-600 text-xs border border-slate-100">{o}</span>
              ))}
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-slate-900">KOL Pipeline</h3>
              <div className="text-xs text-slate-500">{pipeline.length} 位 KOL</div>
            </div>
            <PipelineTable rows={pipeline} />
          </div>
        </div>

        <div className="space-y-6">
          <AiCard title="AI 建议 · Next Actions">
            <ul className="list-disc pl-4 space-y-1">
              <li>TechBro_阿伟 已进入议价阶段 7 天未回复，建议发送「最后跟进」邮件</li>
              <li>LisaFitness 的 Preview 审核已超时，建议一键催稿</li>
              <li>整体预算使用率 64%，建议控制高报价 KOL 比例</li>
            </ul>
            <div className="pt-2 flex gap-2">
              <button className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md bg-brand-600 text-white text-xs font-medium hover:bg-brand-700">
                <Sparkles size={12} /> 生成 Outreach 邮件
              </button>
              <button className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md border border-brand-200 text-brand-700 text-xs font-medium hover:bg-brand-50">
                <Sparkles size={12} /> 生成 PM Summary
              </button>
            </div>
          </AiCard>

          <div className="bg-white rounded-xl border border-slate-200 p-5">
            <h3 className="text-sm font-semibold text-slate-900 mb-3">当前风险</h3>
            <RiskAlert flags={pipeline.flatMap((p) => p.riskFlags)} />
          </div>
        </div>
      </div>
    </div>
  );
}