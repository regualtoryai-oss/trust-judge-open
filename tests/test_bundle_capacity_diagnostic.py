"""A multi-request SDK stage must retain the failed native request capacity."""
import pytest
from app.agents.provider import ModelOutputInvalid
from test_bundle_workbench import BundleFixture, sources, graph


@pytest.mark.parametrize('actual,expected', [(1024, 1024), (4096, 4096), (None, 4096), (-1, 4096)])
async def test_native_failure_ceiling_is_not_replaced_by_stage_maximum(tmp_path, actual, expected):
    class FailedNative(BundleFixture):
        def capacity_policy(self, stage):
            return {'initial_tokens': 4096, 'max_tokens': 4096, 'retries': 0}
        async def select_source_bundles(self, *args, **kwargs):
            raise ModelOutputInvalid('Explicit synthetic budget stop', stage='source_bundle_selection',
                code='whole_episode_budget_exceeded', details={'last_max_output_tokens': actual,
                    'known_tokens': 1203, 'input_estimate': 10679})
    docs, rows = sources()
    with pytest.raises(ModelOutputInvalid) as error:
        await graph(tmp_path, docs, rows, FailedNative(None))
    assert error.value.details['last_max_output_tokens'] == expected
    assert error.value.details['max_output_tokens_limit'] == 4096
    assert error.value.details['capacity_attempts'] == 0
