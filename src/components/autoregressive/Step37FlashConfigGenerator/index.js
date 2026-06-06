import React from 'react';
import ConfigGenerator from '../../base/ConfigGenerator';

/**
 * Step-3.7-Flash Configuration Generator
 *
 * Step-3.7-Flash: 198B-parameter sparse MoE vision-language model from StepFun.
 *   ~11B activated parameters per token.
 *   256K context window.
 *
 * GPU requirements (FP8, ~198 GB weights):
 *   MI300X (192GB): tp=8 ep=8
 *   MI325X (256GB): tp=8 ep=8
 *   MI350X (288GB): tp=8 ep=8
 *   MI355X (288GB): tp=8 ep=8
 */
const Step37FlashConfigGenerator = () => {
  const config = {
    modelFamily: 'Step-3.7-Flash',

    options: {
      hardware: {
        name: 'hardware',
        title: 'Hardware Platform',
        items: [
          { id: 'mi300x', label: 'MI300X', default: true },
          { id: 'mi325x', label: 'MI325X', default: false },
          { id: 'mi350x', label: 'MI350X', default: false },
          { id: 'mi355x', label: 'MI355X', default: false }
        ]
      },
      quantization: {
        name: 'quantization',
        title: 'Quantization',
        items: [
          { id: 'fp8', label: 'FP8', default: true }
        ]
      },
      reasoningParser: {
        name: 'reasoningParser',
        title: 'Reasoning Parser',
        items: [
          { id: 'enabled', label: 'Enabled', default: true },
          { id: 'disabled', label: 'Disabled', default: false }
        ],
        commandRule: (value) => value === 'enabled' ? '--reasoning-parser step3p5' : null
      },
      toolcall: {
        name: 'toolcall',
        title: 'Tool Call Parser',
        items: [
          { id: 'enabled', label: 'Enabled', default: true },
          { id: 'disabled', label: 'Disabled', default: false }
        ],
        commandRule: (value) => value === 'enabled' ? '--tool-call-parser step3p5' : null
      }
    },

    modelConfigs: {
      mi300x: { fp8: { tp: 8, ep: 8 } },
      mi325x: { fp8: { tp: 8, ep: 8 } },
      mi350x: { fp8: { tp: 8, ep: 8 } },
      mi355x: { fp8: { tp: 8, ep: 8 } }
    },

    generateCommand: function (values) {
      const { hardware, quantization } = values;
      const hwConfig = this.modelConfigs[hardware]?.[quantization];
      if (!hwConfig) {
        return '# Please select a valid hardware and quantization combination';
      }

      const modelName = 'stepfun-ai/Step-3.7-Flash-FP8';
      const tpValue = hwConfig.tp;
      const epValue = hwConfig.ep;

      let cmd = 'sglang serve \\\n';
      cmd += `  --model-path ${modelName}`;
      cmd += ` \\\n  --tp ${tpValue}`;
      cmd += ` \\\n  --ep ${epValue}`;
      cmd += ' \\\n  --trust-remote-code';

      Object.entries(this.options).forEach(([key, option]) => {
        if (option.commandRule) {
          const rule = option.commandRule(values[key], values);
          if (rule) {
            cmd += ` \\\n  ${rule}`;
          }
        }
      });

      return cmd;
    }
  };

  return <ConfigGenerator config={config} />;
};

export default Step37FlashConfigGenerator;
