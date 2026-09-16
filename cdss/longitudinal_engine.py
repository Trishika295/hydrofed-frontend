from datetime import datetime

class LongitudinalTrendEngine:
    def __init__(self, prob_change_threshold=0.10, uncertainty_warning_threshold=0.15):
        self.prob_change_threshold = prob_change_threshold
        self.uncertainty_warning_threshold = uncertainty_warning_threshold

    def analyze_trend(self, visits):
        """
        Analyzes a patient's visit history to estimate trend.
        visits: list of dicts containing:
          - 'visit_timestamp': timestamp string or datetime
          - 'pneumonia_probability': float (0.0 to 1.0)
          - 'uncertainty': float (standard deviation from MC Dropout)
          - 'model_version': string
        Returns:
          dict containing:
            trend: 'IMPROVING' | 'STABLE' | 'WORSENING' | 'UNCERTAIN'
            warning: string or None
            message: clinical explanation string
        """
        if not visits or len(visits) < 2:
            return {
                'trend': 'UNCERTAIN',
                'warning': None,
                'message': "Baseline visit recorded. Longitudinal trend estimation requires subsequent visits."
            }

        # Sort visits by timestamp (oldest to newest)
        def parse_time(v):
            t = v.get('visit_timestamp')
            if isinstance(t, str):
                try:
                    return datetime.fromisoformat(t)
                except ValueError:
                    # Try other formats or fallback
                    return datetime.utcnow()
            return t or datetime.utcnow()

        sorted_visits = sorted(visits, key=parse_time)
        
        # Safety Check 1: Model Version Consistency
        versions = {v.get('model_version') for v in sorted_visits if v.get('model_version')}
        if len(versions) > 1:
            return {
                'trend': 'UNCERTAIN',
                'warning': "Model version changed between visits.",
                'message': "Longitudinal probability comparison unavailable due to model-version difference."
            }

        latest_visit = sorted_visits[-1]
        previous_visit = sorted_visits[-2]
        
        latest_prob = latest_visit.get('pneumonia_probability')
        prev_prob = previous_visit.get('pneumonia_probability')
        
        latest_unc = latest_visit.get('uncertainty', 0.0)

        # Safety Check 2: High Uncertainty
        warning = None
        if latest_unc > self.uncertainty_warning_threshold:
            warning = "High prediction uncertainty. Clinician review required."

        if latest_prob is None or prev_prob is None:
            return {
                'trend': 'UNCERTAIN',
                'warning': warning,
                'message': "Incomplete diagnostic records. Probability data missing."
            }

        # Calculate difference: positive difference = worsening, negative = improving
        prob_diff = latest_prob - prev_prob
        
        if prob_diff <= -self.prob_change_threshold:
            trend = 'IMPROVING'
            message = "AI-assisted trend suggests improvement; clinical assessment required."
        elif prob_diff >= self.prob_change_threshold:
            trend = 'WORSENING'
            message = "AI-assisted trend suggests potential progression/worsening; clinician review required."
        else:
            trend = 'STABLE'
            message = "AI-assisted trend suggests stable pneumonia indices; clinical correlation recommended."

        return {
            'trend': trend,
            'warning': warning,
            'message': message,
            'prob_difference': prob_diff
        }
if __name__ == '__main__':
    engine = LongitudinalTrendEngine()
    dummy_visits = [
        {'visit_timestamp': '2026-08-01T10:00:00', 'pneumonia_probability': 0.85, 'uncertainty': 0.05, 'model_version': 'v1.0'},
        {'visit_timestamp': '2026-08-10T12:00:00', 'pneumonia_probability': 0.45, 'uncertainty': 0.04, 'model_version': 'v1.0'}
    ]
    print(engine.analyze_trend(dummy_visits))
