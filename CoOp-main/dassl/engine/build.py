from dassl.utils import Registry, check_availability

TRAINER_REGISTRY = Registry("TRAINER") # Trainer 클래스들을 이름별로 등록할 수 있는 레지스트리


def build_trainer(cfg):
    """cfg → cfg.TRAINER.NAME → 클래스 인스턴스 Trainer(cfg) 리턴"""
    avai_trainers = TRAINER_REGISTRY.registered_names()
    check_availability(cfg.TRAINER.NAME, avai_trainers) # cfg에서 요청한 trainer 이름이 등록되어 있는지 확인 / 없으면 에러 발생
    if cfg.VERBOSE:
        print("Loading trainer: {}".format(cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg) # 이름 → 클래스 → 클래스 인스턴스
