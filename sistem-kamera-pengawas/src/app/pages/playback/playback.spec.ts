import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Playback } from './playback';

describe('Playback', () => {
  let component: Playback;
  let fixture: ComponentFixture<Playback>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Playback]
    })
    .compileComponents();

    fixture = TestBed.createComponent(Playback);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
