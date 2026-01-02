import os
import re
import json
from typing import Dict, Tuple, List, Any
import google.generativeai as genai
import google.generativeai.types as types
import app.config as cfg
from app.action_type import ActionType


class LLMAssistant:
    def __init__(self, generation_config: types.GenerationConfig = None):
        """
        Gemini(Google Generative AI) SDK 설정 및 LLM 보조 파라미터.
        """
        self.api_key = cfg.GEMINI_API_KEY
        self.model_name = cfg.LLM_MODEL
        self.top_k = 5
        self.generation_config = generation_config or types.GenerationConfig(
            temperature=0.2,
            response_mime_type="application/json",
        )
        self.model = None
        self.calls = 0
        self.cache_hits = 0
        self.cache: Dict[Tuple[int, Tuple[int, ...]], Any] = {}
        self._ensure_model()

    def _ensure_model(self):
        """Gemini 모델을 준비합니다."""
        if self.model is not None:
            return
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
            )
        except Exception as e:
            print(f"[LLM Assist] Gemini 모델 준비 실패: {e}")
            self.model = None

    def build_llm_prompt(self, env, task, priority, time_step, action_mask, constraints="", dqn_action=""):
        """
        env: RideSharingEnvironment
        action_mask: np.array(V, A)
        """
        vehicles_desc, requests_desc = env.build_llm_state_snapshot()
        action_space_desc = env.build_llm_action_space()
        mask_text = env.format_action_mask(action_mask)

        system_prompt = (
            "역할: DRT+DQN 보조 에이전트. 상태를 분석해 상위 3개 행동을 추천하거나 평가.\n"
            f"절차: Situation Analysis → Action Suggestion(상위 {self.top_k}) → Reasoning → JSON Output.\n"
            "제약: 불필요 서사 금지, 단문/숫자 중심. action_mask=0 또는 제약 위반 행동 금지.\n"
            "출력은 반드시 JSON 한 덩어리만 반환. JSON 외 추가 텍스트 금지.\n"
            'JSON 스키마: {"summary":"..."},{"top_actions":[{"rank":1,"action":"...","reason":"..."},{"rank":2,"action":"...","reason":"..."},{"rank":3,"action":"...","reason":"..."}],"notes":"..."}\n'
            'summary: 상황 요약\n'
            f'top_actions: 상위 {self.top_k}개 행동 추천\n'
            'notes: 추가 설명\n'
        )

        user_prompt = (
            f"[Task] {task}\n"
            f"[Priority] {priority}\n"
            f"[Time] t={time_step}\n"
            "[Vehicles]\n- " + "\n- ".join(vehicles_desc) + "\n"
            "[Requests]\n- " + ("\n- ".join(requests_desc) if requests_desc else "없음") + "\n"
            "[Action Space]\n"
            f"- actions: {action_space_desc}\n"
            f"- action_mask: {mask_text}\n"
            "[Constraints]\n"
            f"- {constraints if constraints else '기본 용량/시간창/동승 규칙 준수'}\n"
        )
        if task == "evaluate" and dqn_action:
            user_prompt += f"[DQN Action]\n- {dqn_action}\n"
        user_prompt += f"[Output]\n- 상위 {self.top_k}개 추천. JSON만 반환.\n"

        return system_prompt + "\n" + user_prompt

    @staticmethod
    def parse_llm_output(llm_text):
        return json.loads(llm_text)

    def _call_gemini(self, prompt):
        """내장 Gemini 모델로 호출. 실패 시 None 반환."""
        self._ensure_model()
        if self.model is None:
            return None
        try:
            response = self.model.generate_content(prompt)
            if hasattr(response, "text"):
                return response.text
            return str(response)
        except Exception as e:
            print(f"[LLM Assist] Gemini 호출 실패: {e}")
            return None

    def _extract_ids_from_action_text(self, action_text):
        """
        action_text 예시: 'V1 assign R104', 'V0 -> slot2', 'REJECT', 'WAIT'
        반환: (vehicle_idx 또는 None, request_id 또는 None, is_reject: bool)
        """
        text = action_text.upper()
        if "REJECT" in text or "WAIT" in text:
            return None, None, True

        vehicle_idx = None
        request_id = None

        v_match = re.search(r"V(\d+)", text)
        if v_match:
            vehicle_idx = int(v_match.group(1))

        r_match = re.search(r"R(\d+)", text)
        if r_match:
            request_id = int(r_match.group(1))
        else:
            # slot 번호로만 주어진 경우 대응
            slot_match = re.search(r"SLOT\s*(\d+)", text)
            if slot_match:
                request_id = slot_match.group(1)  # 슬롯은 후에 매핑

        return vehicle_idx, request_id, False

    def map_llm_action(self, llm_json, env, action_mask, get_action_info_fn):
        """
        LLM의 top_actions[0]을 (vehicle_idx, action_idx, info)로 매핑.
        매핑 실패 시 None 반환.
        """
        top_actions = llm_json.get("top_actions", [])
        if not top_actions:
            return None

        action_text = top_actions[0].get("action", "")
        v_id, r_id_or_slot, is_reject = self._extract_ids_from_action_text(action_text)

        if is_reject:
            action_idx = cfg.POSSIBLE_ACTION - 1
            # 차량을 지정하지 않았다면 첫 번째 차량으로 폴백
            vehicle_idx = v_id if v_id is not None else 0
        else:
            # 차량/요청이 없으면 실패
            if v_id is None:
                return None

            # 요청 id가 주어졌으면 슬롯으로 변환
            if r_id_or_slot is not None and isinstance(r_id_or_slot, int):
                slot_idx = env.get_request_slot_by_id(r_id_or_slot)
            else:
                # 숫자 문자열(slot)인 경우
                try:
                    slot_idx = int(r_id_or_slot) if r_id_or_slot is not None else None
                except Exception:
                    slot_idx = None

            if slot_idx is None:
                return None

            vehicle_idx = v_id
            action_idx = slot_idx

        # 마스크 체크
        try:
            if action_mask[vehicle_idx][action_idx] != 1:
                return None
        except Exception:
            return None

        return [vehicle_idx, action_idx, get_action_info_fn(vehicle_idx, action_idx)]

    def act_with_llm(
        self,
        env,
        state,
        action_mask,
        get_action_info_fn,
        fallback_fn,
        priority="대기시간 최소화 > 시간창 준수 > 승차율",
        constraints="",
        task="recommend",
        dqn_action="",
    ):
        """
        get_action_info_fn: (v_idx, a_idx) -> info dict
        fallback_fn: 실패 시 호출할 콜러블(인자 없음)
        """
        # 캐시 키: (현재 시간스텝, action_mask flatten)
        try:
            mask_tuple = tuple(int(x) for x in action_mask.flatten().tolist())
            cache_key = (env.curr_time, mask_tuple)
        except Exception:
            cache_key = None

        if cache_key and cache_key in self.cache:
            self.cache_hits += 1
            llm_json = self.cache[cache_key]
            mapped = self.map_llm_action(llm_json, env, action_mask, get_action_info_fn)
            if mapped is not None:
                return mapped

        prompt = self.build_llm_prompt(
            env=env,
            task=task,
            priority=priority,
            time_step=env.curr_time,
            action_mask=action_mask,
            constraints=constraints,
            dqn_action=dqn_action,
        )
        try:
            llm_text = self._call_gemini(prompt)
            if llm_text is None:
                raise ValueError("LLM 응답이 비어있음")
            llm_json = self.parse_llm_output(llm_text)
            self.calls += 1
            if cache_key:
                self.cache[cache_key] = llm_json
            mapped = self.map_llm_action(llm_json, env, action_mask, get_action_info_fn)
            if mapped is not None:
                return mapped
        except Exception as e:
            print(f"[LLM Assist] 사용 실패: {e}")

        # 폴백: 기존 DQN 정책
        return fallback_fn()

